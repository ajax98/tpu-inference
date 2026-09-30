# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Metadata scheduler kernel and SMEM data structures for Batched MLA.

Precomputes memory addresses, DMA offsets, task assignments, and synchronization
wait counters on TPU SMEM before executing the attention compute pipeline.
"""

import dataclasses
import functools
from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.mla.v3 import configs, utils

# --- SMEM Descriptor & Struct Abstractions ---


class FieldOffset:
    """A Python descriptor that generates `.at[pos + offset]` lazy lookups in Pallas."""

    def __init__(self, offset: int | None = None, is_abstract: bool = False):
        if offset is None:
            assert is_abstract
        self.offset = offset
        self.__isabstractmethod__ = is_abstract

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if isinstance(obj.data, (jax.Array, np.ndarray)):
            return obj.data[obj.pos + self.offset]
        return obj.data.at[obj.pos + self.offset]


@dataclasses.dataclass(frozen=True)
class DmaNew(ABC):
    """Base class for new-token DMA struct entries in SMEM."""
    data: Any
    pos: Any

    fetch_hbm = FieldOffset(is_abstract=True)
    fetch_vmem = FieldOffset(is_abstract=True)
    wb_hbm = FieldOffset(is_abstract=True)
    wb_vmem = FieldOffset(is_abstract=True)
    _flags = FieldOffset(is_abstract=True)

    @staticmethod
    @abstractmethod
    def num_fields() -> int:
        ...

    @abstractmethod
    def set_flags(self, fetch_val, wb_val):
        ...

    @property
    @abstractmethod
    def fetch_val(self):
        ...

    @property
    @abstractmethod
    def wb_val(self):
        ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SeqAlongLaneDmaNew(DmaNew):
    """Descriptor for new-token DMA in SEQ_ALONG_LANE layout (5 fields).

  Fetch: the new tokens are contiguous in HBM and in VMEM, so entry 0 carries
  one coalesced DMA for all of them and the other entries stay zero.
  Writeback: one DMA per page, since `page_indices` scatters the destinations.
  `wb_hbm` packs `(physical_page << page_size_log2) | offset_in_page`.
  `_flags` packs the fetch and writeback token counts (multiples of 128) in
  its low and high 16 bits.
  """
    fetch_hbm = FieldOffset(0)
    fetch_vmem = FieldOffset(1)
    wb_hbm = FieldOffset(2)
    wb_vmem = FieldOffset(3)
    _flags = FieldOffset(4)

    @staticmethod
    def num_fields() -> int:
        return 5

    @property
    def fetch_val(self):
        val = self._flags
        if hasattr(val, "get"):
            val = val.get()
        return val & 0xFFFF

    @property
    def wb_val(self):
        val = self._flags
        if hasattr(val, "get"):
            val = val.get()
        return val >> 16

    def set_flags(self, fetch_val, wb_val):
        self._flags[...] = fetch_val | (wb_val << 16)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemWrapper:
    """Maps physical 1D SMEM buffer to logical multi-dimensional indexing."""
    data: Any
    shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create_shape_dtype(cls, shape):
        return cls(data=jax.ShapeDtypeStruct((np.prod(shape), ), jnp.int32),
                   shape=shape)

    def _get_pos(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices, )
        strides = pl.strides_from_shape(self.shape)
        assert len(strides) == len(indices)

        pos = 0
        for stride, idx in zip(strides, indices):
            pos += stride * idx
        return pos

    def __getitem__(self, indices):
        return self.data[self._get_pos(indices)]

    def __setitem__(self, indices, value):
        self.data[self._get_pos(indices)] = value


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemArrayOfStructs(SmemWrapper):
    """Wraps physical 1D SMEM buffer as a multi-dimensional array of typed structs."""
    struct_cls: type[DmaNew] = dataclasses.field(metadata=dict(static=True))
    struct_size: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create_shape_dtype(cls, shape, struct_cls, struct_size):  # pytype: disable=bad-override
        assert struct_size == struct_cls.num_fields()
        return cls(
            data=jax.ShapeDtypeStruct((np.prod(shape) * struct_size, ),
                                      jnp.int32),
            shape=shape,
            struct_cls=struct_cls,
            struct_size=struct_size,
        )

    def _get_pos(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices, )
        strides = pl.strides_from_shape(self.shape)
        assert len(strides) == len(indices)

        pos = 0
        for stride, idx in zip(strides, indices):
            pos += stride * idx
        return pos * self.struct_size

    def __getitem__(self, indices):
        pos_start = self._get_pos(indices)
        return self.struct_cls(self.data, pos_start)


# `actual_steps` is padded to one 128-word SMEM tile. It is the only schedule
# leaf kept in SMEM (see `MlaSchedule.in_specs`/`out_specs`): the schedule kernel
# writes it straight into an SMEM output and the attention kernel takes that
# buffer as an SMEM operand, so reading it costs no HBM->SMEM copy.
ACTUAL_STEPS_WORDS = 128


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MlaSchedule:
    """Schedule pytree containing all precomputed metadata arrays."""

    s_idx: SmemWrapper  # [steps, batch]
    q_idx: SmemWrapper  # [steps, batch]
    k_idx: SmemWrapper  # [steps, batch]
    is_last_k: SmemWrapper  # [steps, batch]
    do_writeback: SmemWrapper  # [steps, batch]
    dma_q: SmemWrapper  # [steps, batch, 2]
    dma_kv_cache: SmemWrapper  # [steps, batch, bkv_p_cache, 2]
    dma_kv_new: SmemArrayOfStructs  # [steps, batch, bkv_p_new]
    total_wait_kv_in: SmemWrapper  # [steps]
    total_wait_kv_out: SmemWrapper  # [steps]
    total_wait_q_in: SmemWrapper  # [steps]
    total_wait_o_out: SmemWrapper  # [steps]
    actual_steps: jax.Array  # [ACTUAL_STEPS_WORDS], only word 0 is used.

    cfgs: configs.MlaConfigs = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create_shape_dtype(cls, cfgs: configs.MlaConfigs, multiplier: int = 1):
        effective_max_steps = cfgs.max_steps_ub * multiplier

        idx_wrapper = SmemWrapper.create_shape_dtype(
            (effective_max_steps, cfgs.batch_size))
        steps_wrapper = SmemWrapper.create_shape_dtype((effective_max_steps, ))

        return cls(
            s_idx=idx_wrapper,
            q_idx=idx_wrapper,
            k_idx=idx_wrapper,
            is_last_k=idx_wrapper,
            do_writeback=idx_wrapper,
            dma_q=SmemWrapper.create_shape_dtype(
                (effective_max_steps, cfgs.batch_size, 2)),
            dma_kv_cache=SmemWrapper.create_shape_dtype(
                (effective_max_steps, cfgs.batch_size,
                 max(1, cfgs.bkv_p_cache), 2)),
            dma_kv_new=SmemArrayOfStructs.create_shape_dtype(
                (effective_max_steps, cfgs.batch_size, cfgs.bkv_p_new),
                struct_cls=SeqAlongLaneDmaNew,
                struct_size=cfgs.dma_kv_new_size,
            ),
            total_wait_kv_in=steps_wrapper,
            total_wait_kv_out=steps_wrapper,
            total_wait_q_in=steps_wrapper,
            total_wait_o_out=steps_wrapper,
            actual_steps=jax.ShapeDtypeStruct((ACTUAL_STEPS_WORDS, ),
                                              jnp.int32),  # pytype: disable=bad-argument-type
            cfgs=cfgs,
        )

    def step_leaves(self) -> list[Any]:
        """Leaves indexed by step and stored in HBM: all but `actual_steps`."""
        return jax.tree_util.tree_leaves(
            dataclasses.replace(self, actual_steps=None))

    @property
    def total_hbm_words(self) -> int:
        return sum(int(leaf.shape[0]) for leaf in self.step_leaves())

    def get_dma_kv_cache(
        self,
        step: jax.typing.ArrayLike,
        batch_idx: jax.typing.ArrayLike,
        page_idx: jax.typing.ArrayLike,
    ) -> tuple[jax.Array, jax.Array]:
        pos = self.dma_kv_cache._get_pos((step, batch_idx, page_idx, 0))
        src_off = self.dma_kv_cache.data[pos]
        sz = self.dma_kv_cache.data[pos + 1]
        return src_off, sz

    def get_dma_q(
            self, step: jax.typing.ArrayLike,
            batch_idx: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array]:
        pos = self.dma_q._get_pos((step, batch_idx, 0))
        src_hbm = self.dma_q.data[pos]
        sz = self.dma_q.data[pos + 1]
        return src_hbm, sz

    def scratch_shapes(self):
        return jax.tree.map(lambda x: pltpu.SMEM(x.shape, x.dtype), self)

    def memory_spaces(self):
        """HBM for the per-step leaves, SMEM for `actual_steps`."""
        spaces = jax.tree.map(lambda _: pltpu.HBM, self)
        return dataclasses.replace(spaces, actual_steps=pltpu.SMEM)

    def in_specs(self):
        return jax.tree.map(lambda space: pl.BlockSpec(memory_space=space),
                            self.memory_spaces())

    def out_specs(self):
        return self.in_specs()

    def out_shape(self):
        return jax.tree.map(lambda x, space: space(x.shape, x.dtype), self,
                            self.memory_spaces())


# --- Scheduler Computation Core ---


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LoopCarry:
    hbm_offset: jax.Array | int
    count: jax.Array | int


def _mask_out_steps(
    step: jax.typing.ArrayLike,
    schedule_smem: MlaSchedule,
    b_idx: jax.typing.ArrayLike,
):
    """Masks out unused schedule entries at (step, b_idx)."""
    schedule_smem.s_idx[step, b_idx] = -1
    schedule_smem.q_idx[step, b_idx] = 0
    schedule_smem.k_idx[step, b_idx] = 0
    schedule_smem.is_last_k[step, b_idx] = 0
    schedule_smem.do_writeback[step, b_idx] = 0
    schedule_smem.dma_q[step, b_idx, 0] = 0
    schedule_smem.dma_q[step, b_idx, 1] = 0

    for i in range(schedule_smem.cfgs.bkv_p_cache):
        schedule_smem.dma_kv_cache[step, b_idx, i, 0] = 0
        schedule_smem.dma_kv_cache[step, b_idx, i, 1] = 0

    for i in range(schedule_smem.cfgs.bkv_p_new):
        dma_entry = schedule_smem.dma_kv_new[step, b_idx, i]
        dma_entry.fetch_hbm[...] = 0
        dma_entry.fetch_vmem[...] = 0
        dma_entry.wb_hbm[...] = 0
        dma_entry.wb_vmem[...] = 0
        dma_entry.set_flags(0, 0)

    schedule_smem.total_wait_kv_in[step] = 0
    schedule_smem.total_wait_kv_out[step] = 0
    schedule_smem.total_wait_q_in[step] = 0
    schedule_smem.total_wait_o_out[step] = 0


def _write_schedule_to_hbm(
    schedule_smem: MlaSchedule,
    schedule_hbm: MlaSchedule,
    hbm_offset: jax.typing.ArrayLike,
    num_steps: jax.typing.ArrayLike,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.MlaConfigs,
):
    """Writes num_steps from schedule_smem to schedule_hbm.

  Every flush advances `hbm_offset` by a full `max_steps_ub`, so the write can
  only stay in bounds if the buffer is sized for the worst case. That is
  `configs.MlaConfigs.max_schedule_size_multiplier`, which raises the caller's
  multiplier to cover `max_steps_needed` -- a static upper bound over every
  ragged split the shapes admit. Callers must not flush an empty buffer, which
  would advance past the last real step.

  Only the steps the flush holds are copied. SMEM->HBM DMAs are slow, and always
  writing the whole `max_steps_ub` buffer made a one-step decode schedule move
  ~300 KiB. `num_steps` is traced, but a dynamic-size slice cannot be discharged
  in interpret mode, so the copy is a traced number of fixed-size chunks rather
  than one variable-size DMA. 128-step chunks keep every chunk offset a multiple
  of the 128-word DMA tile whatever a leaf's words-per-step. Because
  `max_steps_ub` is a multiple of 128, whole chunks stay inside both the SMEM
  buffer and this flush's HBM slot. When `max_steps_needed` bounds every flush
  to one chunk, the chunk shrinks to fit and the loop disappears.
  """
    hbm_offset_aligned = pl.multiple_of(hbm_offset, 128)  # pytype: disable=bad-argument-type
    flat_hbm = schedule_hbm.step_leaves()
    flat_smem = schedule_smem.step_leaves()

    # At least one step: the final flush is traced even for a zero-token shape,
    # where `max_steps_needed` is 0, although `count > 0` never lets it run.
    max_flush_steps = max(1, min(cfgs.max_steps_ub, cfgs.max_steps_needed))
    chunk_steps = min(128, max_flush_steps)
    max_chunks = pl.cdiv(max_flush_steps, chunk_steps)

    def chunk_copies(chunk_idx):
        copies = []
        for h, s in zip(flat_hbm, flat_smem):
            element_size = s.shape[0] // cfgs.max_steps_ub
            chunk_size = utils.align_to(chunk_steps * element_size, 128)
            src_offset = chunk_idx * chunk_size
            if not isinstance(src_offset, int):
                src_offset = pl.multiple_of(src_offset, 128)
            dst_offset = pl.multiple_of(
                hbm_offset_aligned * element_size + src_offset, 128)
            copies.append(
                pltpu.make_async_copy(
                    s.at[pl.ds(src_offset, chunk_size)],
                    h.at[pl.ds(dst_offset, chunk_size)],
                    dma_sem.at[0],
                ))
        return copies

    if max_chunks == 1:
        copies = chunk_copies(0)
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()
    else:
        num_chunks = pl.cdiv(jnp.asarray(num_steps), chunk_steps)

        @pl.loop(0, num_chunks)
        def _(chunk_idx):
            for copy in chunk_copies(chunk_idx):
                copy.start()

        @pl.loop(0, num_chunks)
        def _(chunk_idx):
            for copy in chunk_copies(chunk_idx):
                copy.wait()


def _compute_waits(
    schedule: MlaSchedule,
    start_step: jax.typing.ArrayLike,
    end_step: jax.typing.ArrayLike,
    *,
    cfgs: configs.MlaConfigs,
):
    """Computes total wait lane counts for DMA synchronization."""

    @jax.named_scope("compute_waits")
    def body(step, _):
        # KV IN: every descriptor carries its token count (a lane-tile multiple).
        kv_in_tokens = 0
        for b in range(cfgs.batch_size):
            for i in range(cfgs.bkv_p_cache):
                _, sz = schedule.get_dma_kv_cache(step, b, i)
                kv_in_tokens += sz
            for i in range(cfgs.bkv_p_new):
                kv_in_tokens += schedule.dma_kv_new[step, b, i].fetch_val

        schedule.total_wait_kv_in[step] = kv_in_tokens

        # KV OUT
        kv_out_tokens = 0
        for b in range(cfgs.batch_size):
            do_writeback = schedule.do_writeback[step, b] == 1
            for i in range(cfgs.bkv_p_new):
                dma_entry = schedule.dma_kv_new[step, b, i]
                kv_out_tokens += jnp.where(do_writeback, dma_entry.wb_val, 0)

        schedule.total_wait_kv_out[step] = kv_out_tokens

        # Q IN
        q_in_tokens = 0
        for b in range(cfgs.batch_size):
            _, q_sz = schedule.get_dma_q(step, b)
            q_in_tokens += q_sz
        schedule.total_wait_q_in[step] = q_in_tokens

        # O OUT
        o_out_tokens = 0
        for b in range(cfgs.batch_size):
            is_last_k = schedule.is_last_k[step, b] == 1
            _, q_sz = schedule.get_dma_q(step, b)
            o_out_tokens += jnp.where(is_last_k, q_sz, 0)
        schedule.total_wait_o_out[step] = o_out_tokens

    jax.lax.fori_loop(start_step, end_step, body, None)


def flush_to_hbm(
    count: jax.typing.ArrayLike,
    schedule: MlaSchedule,
    schedule_hbm_ref: MlaSchedule,
    hbm_offset: jax.typing.ArrayLike,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.MlaConfigs,
):
    last_step = utils.align_to(count, cfgs.batch_size)
    num_steps = last_step // cfgs.batch_size

    @pl.loop(count, last_step)
    @jax.named_scope("mask_out_steps")
    def body(idx):
        step, b_idx = divmod(idx, cfgs.batch_size)
        _mask_out_steps(step, schedule, b_idx)

    _compute_waits(schedule, 0, num_steps, cfgs=cfgs)
    _write_schedule_to_hbm(
        schedule,
        schedule_hbm_ref,
        hbm_offset,
        num_steps,
        dma_sem,
        cfgs=cfgs,
    )
    return hbm_offset + cfgs.max_steps_ub, 0


def compute_metadata(
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    page_indices_ref: jax.Ref,
    distribution_ref: jax.Ref,
    schedule: MlaSchedule,
    schedule_hbm_ref: MlaSchedule,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.MlaConfigs,
) -> LoopCarry:
    """Populates schedule metadata using native TPU jax.lax.fori_loop."""

    @jax.named_scope("k_loop")
    def k_loop(
        k_idx,
        carry: LoopCarry,
        *,
        s_idx,
        q_idx,
        q_end,
        q_src,
        q_sz_task,
        k_len,
        q_len,
        end_k_idx,
    ):
        count = carry.count
        step, target_lane = divmod(count, cfgs.batch_size)

        schedule.s_idx[step, target_lane] = s_idx
        schedule.q_idx[step, target_lane] = q_idx
        schedule.k_idx[step, target_lane] = k_idx

        is_last_k = jnp.where(k_idx == end_k_idx - 1, 1, 0)
        schedule.is_last_k[step, target_lane] = is_last_k

        schedule.dma_q[step, target_lane, 0] = q_src
        schedule.dma_q[step, target_lane, 1] = q_sz_task

        kv_len_start = k_idx * cfgs.bkv_sz
        kv_p_start = k_idx * cfgs.bkv_p
        kv_left = k_len - kv_len_start
        kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        p_offset = s_idx * cfgs.serve.pages_per_seq + kv_p_start

        for i in range(cfgs.bkv_p_cache):
            dst_vmem = i << cfgs.serve.page_size_log2
            dma_sz = jnp.clip(kv_left_frm_cache - dst_vmem, 0,
                              cfgs.serve.page_size)
            src_hbm_idx = jnp.minimum(p_offset + i,
                                      cfgs.serve.num_page_indices - 1)
            hbm_p_idx = page_indices_ref[src_hbm_idx]

            schedule.dma_kv_cache[step, target_lane, i, 0] = hbm_p_idx
            # Token count rounded up to the lane tile: only the tiles holding cached
            # tokens are fetched, not the whole page. Both ends are static: the
            # source starts at offset 0 of the page, the destination at
            # i * page_size.
            schedule.dma_kv_cache[step, target_lane, i,
                                  1] = utils.align_to(dma_sz,
                                                      cfgs.kv_token_align)

        kv_left_frm_new = kv_left - kv_left_frm_cache
        bkv_sz_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
        new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_frm_new)

        q_wb = jnp.maximum(0, (kv_len_start - (k_len - q_len))) // cfgs.bq_sz
        do_writeback = jnp.where((new_sz > 0) & (q_idx == q_wb), 1, 0)
        schedule.do_writeback[step, target_lane] = do_writeback

        def fill_dma_kv_new(i, dst_vmem, dma_sz):
            """Fills new-KV entry `i`: writes back `dma_sz` tokens at `dst_vmem`."""
            dma_entry = schedule.dma_kv_new[step, target_lane, i]
            align = cfgs.kv_token_align
            # Fetch: the new tokens are contiguous in HBM and in VMEM, so one DMA
            # moves all of them. Entry 0 carries it, widened to the lane tile on both
            # ends, and lands right after the cached tiles; the stitch later shifts
            # the tokens down to `bkv_sz_cache`. The other entries stay zero.
            if i == 0:
                src_hbm = q_end - kv_left_frm_new
                src, _, fetch_val = utils.align_span(src_hbm, new_sz, align)
                dma_entry.fetch_hbm[...] = src
                dma_entry.fetch_vmem[...] = utils.align_to(bkv_sz_cache, align)
                fetch_val = jnp.where(new_sz > 0, fetch_val, 0)
            else:
                dma_entry.fetch_hbm[...] = 0
                dma_entry.fetch_vmem[...] = 0
                fetch_val = 0

            # Writeback: one DMA per page, since page_indices scatters the pages,
            # widened to the lane tiles that hold the tokens. The tiles' leading
            # lanes are cached tokens that VMEM holds unchanged, and the trailing
            # ones lie past the sequence end, so rewriting them is harmless.
            tok_idx = kv_len_start + dst_vmem
            p_idx = jnp.minimum(tok_idx >> cfgs.serve.page_size_log2,
                                cfgs.serve.pages_per_seq - 1)
            dst_hbm_idx = jnp.minimum(
                s_idx * cfgs.serve.pages_per_seq + p_idx,
                cfgs.serve.num_page_indices - 1,
            )
            hbm_p_idx = page_indices_ref[dst_hbm_idx]
            p_off = tok_idx & cfgs.serve.page_size_mask
            dst, lead, wb_val = utils.align_span(p_off, dma_sz, align)

            dma_entry.wb_hbm[...] = (
                hbm_p_idx << cfgs.serve.page_size_log2) | dst
            dma_entry.wb_vmem[...] = dst_vmem - lead
            dma_entry.set_flags(fetch_val, jnp.where(dma_sz > 0, wb_val, 0))

        if cfgs.one_new_token:
            assert cfgs.bkv_p_new == 1
            fill_dma_kv_new(0, bkv_sz_cache, new_sz)
        else:
            iters = max(cfgs.bkv_p, cfgs.bkv_p_new)
            for i in range(iters):
                slot_start = i * cfgs.serve.page_size
                slot_end = slot_start + cfgs.serve.page_size

                dst_vmem = jnp.maximum(slot_start, bkv_sz_cache)
                end_in_slot = jnp.minimum(slot_end, bkv_sz_cache + new_sz)
                dma_sz = jnp.maximum(0, end_in_slot - dst_vmem)

                fill_dma_kv_new(i, dst_vmem, dma_sz)

        def flush(carry: LoopCarry):
            hbm_offset = carry.hbm_offset
            new_hbm_offset, _ = flush_to_hbm(
                carry.count,
                schedule,
                schedule_hbm_ref,
                hbm_offset,
                dma_sem,
                cfgs=cfgs,
            )
            return LoopCarry(new_hbm_offset, 0)

        hbm_offset = carry.hbm_offset
        new_count = count + 1

        return jax.lax.cond(
            new_count < cfgs.max_steps_ub * cfgs.batch_size,
            lambda carry: carry,
            flush,
            LoopCarry(hbm_offset, new_count),
        )

    @jax.named_scope("q_loop")
    def q_loop(q_idx, carry, *, s_idx, q_start, q_end, k_len, q_len, num_k):
        q_src = q_start + q_idx * cfgs.bq_sz
        q_sz_task = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)

        start_k_idx = 0
        if (sliding_window := cfgs.model.sliding_window) is not None:
            sw_start_idx = k_len - q_len + q_idx * cfgs.bq_sz - sliding_window + 1
            start_k_idx = jnp.maximum(0, sw_start_idx) // cfgs.bkv_sz

        end_k_idx_causal = (k_len - q_len + q_idx * cfgs.bq_sz + q_sz_task -
                            1) // cfgs.bkv_sz + 1
        end_k_idx = jnp.minimum(num_k, end_k_idx_causal)

        k_loop_fn = functools.partial(
            k_loop,
            s_idx=s_idx,
            q_idx=q_idx,
            q_end=q_end,
            q_src=q_src,
            q_sz_task=q_sz_task,
            k_len=k_len,
            q_len=q_len,
            end_k_idx=end_k_idx,
        )
        return jax.lax.fori_loop(start_k_idx, end_k_idx, k_loop_fn, carry)

    @jax.named_scope("seq_loop")
    def seq_loop(s_idx, carry):
        q_start = cu_q_lens_ref[s_idx]
        q_end = cu_q_lens_ref[s_idx + 1]
        k_len = kv_lens_ref[s_idx]
        q_len = q_end - q_start

        num_q = pl.cdiv(q_len, cfgs.bq_sz)
        num_k = pl.cdiv(k_len, cfgs.bkv_sz)

        q_loop_fn = functools.partial(
            q_loop,
            s_idx=s_idx,
            q_start=q_start,
            q_end=q_end,
            k_len=k_len,
            q_len=q_len,
            num_k=num_k,
        )
        return jax.lax.fori_loop(0, num_q, q_loop_fn, carry)

    start_seq_idx, end_seq_idx = cfgs.mode.get_range(distribution_ref)  # pytype: disable=bad-argument-type
    init_carry = LoopCarry(0, 0)
    return jax.lax.fori_loop(start_seq_idx, end_seq_idx, seq_loop, init_carry)


def rpa_metadata_schedule_kernel(
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    page_indices_ref: jax.Ref,
    distribution_ref: jax.Ref,
    schedule_hbm_ref: MlaSchedule,
    schedule_ref: MlaSchedule,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.MlaConfigs,
):
    """Generates HBM-to-VMEM DMA schedule for Batched MLA."""
    loop_carry = compute_metadata(
        cu_q_lens_ref,
        kv_lens_ref,
        page_indices_ref,
        distribution_ref,
        schedule_ref,
        schedule_hbm_ref,
        dma_sem,
        cfgs=cfgs,
    )
    count = loop_carry.count
    hbm_offset = loop_carry.hbm_offset
    steps = pl.cdiv(count, cfgs.batch_size) + hbm_offset
    # `actual_steps` is an SMEM output, so a scalar store is all it takes.
    schedule_hbm_ref.actual_steps[0] = steps  # pytype: disable=unsupported-operation

    # An empty buffer has nothing to flush, and flushing it anyway would advance
    # `hbm_offset` by another `max_steps_ub` -- past the end of the HBM schedule
    # whenever the step count landed exactly on a buffer boundary.
    @pl.when(count > 0)
    def _():
        flush_to_hbm(
            count,
            schedule_ref,
            schedule_hbm_ref,
            hbm_offset,
            dma_sem,
            cfgs=cfgs,
        )


def generate_mla_metadata(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    distribution: jax.Array,
    cfgs: configs.MlaConfigs,
    *,
    interpret: bool = False,
) -> MlaSchedule:
    """Entry point for generating Batched MLA schedule metadata."""
    schedule_shaped_dtype = MlaSchedule.create_shape_dtype(cfgs)
    schedule_hbm = MlaSchedule.create_shape_dtype(
        cfgs, multiplier=cfgs.max_schedule_size_multiplier)
    # `out_specs` only types the refs inside the kernel; where XLA places the
    # output buffers comes from the `out_shape` avals. Plain ShapeDtypeStructs
    # carry no memory space, so XLA was free to put the schedule in VMEM and then
    # evict it to HBM with copy-start/done pairs before the attention kernel ran.
    # Tagging each leaf pins the per-step leaves in HBM and `actual_steps` in SMEM.
    out_shape = schedule_hbm.out_shape()

    return pl.pallas_call(
        functools.partial(rpa_metadata_schedule_kernel, cfgs=cfgs),
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[],
            out_specs=schedule_hbm.out_specs(),
            scratch_shapes=[
                schedule_shaped_dtype.scratch_shapes(),
                pltpu.SemaphoreType.DMA((1, )),
            ],
        ),
        interpret=interpret,
        name="mla_metadata_schedule",
    )(cu_q_lens, kv_lens, page_indices, distribution)
