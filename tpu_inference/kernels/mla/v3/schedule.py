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

from abc import ABC, abstractmethod
import dataclasses
import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.mla.v3 import configs
from tpu_inference.kernels.mla.v3 import utils


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
  """Descriptor for new-token DMA in SEQ_ALONG_LANE layout (5 fields)."""
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
    return val & 1

  @property
  def wb_val(self):
    val = self._flags
    if hasattr(val, "get"):
      val = val.get()
    return (val >> 1) & 1

  def set_flags(self, fetch_val, wb_val):
    self._flags[...] = fetch_val | (wb_val << 1)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemWrapper:
  """Maps physical 1D SMEM buffer to logical multi-dimensional indexing."""
  data: Any
  shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create_shape_dtype(cls, shape):
    return cls(
        data=jax.ShapeDtypeStruct((np.prod(shape),), jnp.int32), shape=shape
    )

  def _get_pos(self, indices):
    if not isinstance(indices, tuple):
      indices = (indices,)
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
        data=jax.ShapeDtypeStruct((np.prod(shape) * struct_size,), jnp.int32),
        shape=shape,
        struct_cls=struct_cls,
        struct_size=struct_size,
    )

  def _get_pos(self, indices):
    if not isinstance(indices, tuple):
      indices = (indices,)
    strides = pl.strides_from_shape(self.shape)
    assert len(strides) == len(indices)

    pos = 0
    for stride, idx in zip(strides, indices):
      pos += stride * idx
    return pos * self.struct_size

  def __getitem__(self, indices):
    pos_start = self._get_pos(indices)
    return self.struct_cls(self.data, pos_start)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MlaSchedule:
  """Schedule pytree containing all precomputed metadata arrays."""

  s_idx: SmemWrapper  # [steps, batch]
  q_idx: SmemWrapper  # [steps, batch]
  k_idx: SmemWrapper  # [steps, batch]
  is_last_k: SmemWrapper  # [steps, batch]
  do_writeback: SmemWrapper  # [steps, batch]
  # How many unpaged new tokens land in this block. The kernel gates its
  # new-KV merge on this every grid step and used to re-derive it from
  # kv_lens/cu_q_lens indirections plus ~8 ops per lane; `k_loop` already has
  # it while building descriptors.
  new_sz: SmemWrapper  # [steps, batch]
  dma_q: SmemWrapper  # [steps, batch, 2]
  dma_kv_cache: SmemWrapper  # [steps, batch, bkv_p_cache, 3]
  dma_kv_new: SmemArrayOfStructs  # [steps, batch, bkv_p_new]
  total_wait_kv_in: SmemWrapper  # [steps]
  total_wait_kv_out: SmemWrapper  # [steps]
  total_wait_q_in: SmemWrapper  # [steps]
  total_wait_o_out: SmemWrapper  # [steps]
  actual_steps: jax.Array  # [1]

  cfgs: configs.MlaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create_shape_dtype(cls, cfgs: configs.MlaConfigs, multiplier: int = 1):
    effective_max_steps = cfgs.max_steps_ub * multiplier

    idx_wrapper = SmemWrapper.create_shape_dtype(
        (effective_max_steps, cfgs.batch_size)
    )
    steps_wrapper = SmemWrapper.create_shape_dtype((effective_max_steps,))

    return cls(
        s_idx=idx_wrapper,
        q_idx=idx_wrapper,
        k_idx=idx_wrapper,
        is_last_k=idx_wrapper,
        do_writeback=idx_wrapper,
        new_sz=idx_wrapper,
        dma_q=SmemWrapper.create_shape_dtype(
            (effective_max_steps, cfgs.batch_size, 2)
        ),
        dma_kv_cache=SmemWrapper.create_shape_dtype(
            (effective_max_steps, cfgs.batch_size, max(1, cfgs.bkv_p_cache), 2)
        ),
        dma_kv_new=SmemArrayOfStructs.create_shape_dtype(
            (effective_max_steps, cfgs.batch_size, cfgs.bkv_p_new),
            struct_cls=SeqAlongLaneDmaNew,
            struct_size=cfgs.dma_kv_new_size,
        ),
        total_wait_kv_in=steps_wrapper,
        total_wait_kv_out=steps_wrapper,
        total_wait_q_in=steps_wrapper,
        total_wait_o_out=steps_wrapper,
        actual_steps=jax.ShapeDtypeStruct((1,), jnp.int32),  # pytype: disable=bad-argument-type
        cfgs=cfgs,
    )

  def get_dma_kv_cache(
      self,
      step: jax.typing.ArrayLike,
      batch_idx: jax.typing.ArrayLike,
      page_idx: jax.typing.ArrayLike,
  ) -> tuple[jax.Array, jax.Array]:
    """Source page index and validity flag for one cached-KV page.

    Two things combined here. The VMEM destination is not stored: it is
    `page_idx * page_size`, a compile-time constant that the caller's loop
    index supplies, so it does not need a third field read back on every grid
    step (batch_size * bkv_p_cache times, 12 at batch=4, bkv=3). And the reads
    resolve the position once via `_get_pos` and index `data` directly, rather
    than repeating the stride arithmetic per field -- that part is from the
    upstream branch.
    """
    pos = self.dma_kv_cache._get_pos((step, batch_idx, page_idx, 0))
    src_off = self.dma_kv_cache.data[pos]
    sz = self.dma_kv_cache.data[pos + 1]
    return src_off, sz

  def get_dma_q(
      self, step: jax.typing.ArrayLike, batch_idx: jax.typing.ArrayLike
  ) -> tuple[jax.Array, jax.Array]:
    pos = self.dma_q._get_pos((step, batch_idx, 0))
    src_hbm = self.dma_q.data[pos]
    sz = self.dma_q.data[pos + 1]
    return src_hbm, sz

  def scratch_shapes(self):
    return jax.tree.map(lambda x: pltpu.SMEM(x.shape, x.dtype), self)

  def in_specs(self):
    def wrapper(x):
      if x.size == 1:
        return pl.BlockSpec(memory_space=pltpu.SMEM)
      return pl.BlockSpec(memory_space=pltpu.HBM)
    return jax.tree.map(wrapper, self)

  def out_specs(self):
    return jax.tree.map(lambda x: pl.BlockSpec(memory_space=pltpu.HBM), self)

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
  schedule_smem.new_sz[step, b_idx] = 0
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
  """
  hbm_offset_aligned = pl.multiple_of(hbm_offset, 128)  # pytype: disable=bad-argument-type
  flat_hbm = jax.tree_util.tree_leaves(schedule_hbm)
  flat_smem = jax.tree_util.tree_leaves(schedule_smem)
  dma_list = []
  for h, s in zip(flat_hbm, flat_smem):
    element_size = s.shape[0] // cfgs.max_steps_ub
    if h.shape[0] > 1:
      write_size = num_steps * element_size
      write_size = utils.align_to(write_size, 128)
      output_offset = hbm_offset_aligned * element_size
    else:
      write_size = h.shape[0]
      output_offset = 0

    copy = pltpu.make_async_copy(
        s.at[pl.ds(0, write_size)],
        h.at[pl.ds(output_offset, write_size)],
        dma_sem.at[0],
    )
    dma_list.append(copy)

  jax.tree.map(lambda x: x.start(), dma_list)
  jax.tree.map(lambda x: x.wait(), dma_list)


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
    # KV IN
    kv_in_tokens = 0
    for b in range(cfgs.batch_size):
      # SEQ_ALONG_LANE transfers whole pages, so the descriptors carry a
      # validity flag rather than a size and each valid one is `page_size`.
      for i in range(cfgs.bkv_p_cache):
        _, dma_valid = schedule.get_dma_kv_cache(step, b, i)
        kv_in_tokens += dma_valid * cfgs.serve.page_size
      for i in range(cfgs.bkv_p_new):
        dma_entry = schedule.dma_kv_new[step, b, i]
        kv_in_tokens += jnp.where(
            dma_entry.fetch_val > 0, cfgs.serve.page_size, 0
        )

    # Stored as a *token* count, which is what `bref_override.wait_in` needs
    # to size its wait region. It previously held a DMA-chunk count and was
    # never read by anything, while the waiter recomputed the same sum from 16
    # SMEM descriptors on every grid step.
    schedule.total_wait_kv_in[step] = kv_in_tokens

    # KV OUT
    kv_out_tokens = 0
    for b in range(cfgs.batch_size):
      do_writeback = schedule.do_writeback[step, b] == 1
      for i in range(cfgs.bkv_p_new):
        dma_entry = schedule.dma_kv_new[step, b, i]
        kv_out_tokens += jnp.where(
            do_writeback & (dma_entry.wb_val > 0), cfgs.serve.page_size, 0
        )

    schedule.total_wait_kv_out[step] = kv_out_tokens

    # Q IN
    q_in_tokens = 0
    for b in range(cfgs.batch_size):
      _, q_sz = schedule.get_dma_q(step, b)
      q_in_tokens += q_sz
    # Token counts, as for the KV totals: that is what the waiters need, and
    # keeping the `* const` form lets Mosaic see the alignment (see the note in
    # `BatchingQRef.wait_in`).
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

  @pl.loop(count, last_step)
  @jax.named_scope("mask_out_steps")
  def body(idx):
    step, b_idx = divmod(idx, cfgs.batch_size)
    _mask_out_steps(step, schedule, b_idx)

  _compute_waits(schedule, 0, last_step // cfgs.batch_size, cfgs=cfgs)
  _write_schedule_to_hbm(
      schedule,
      schedule_hbm_ref,
      hbm_offset,
      cfgs.max_steps_ub,
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
      dma_sz = jnp.clip(kv_left_frm_cache - dst_vmem, 0, cfgs.serve.page_size)
      src_hbm_idx = jnp.minimum(p_offset + i, cfgs.serve.num_page_indices - 1)
      hbm_p_idx = page_indices_ref[src_hbm_idx]

      schedule.dma_kv_cache[step, target_lane, i, 0] = hbm_p_idx
      # Whole-page transfer: the second field is a validity flag, not a size.
      # The VMEM destination is `i * page_size` and is recomputed by the
      # reader rather than stored; see `get_dma_kv_cache`.
      schedule.dma_kv_cache[step, target_lane, i, 1] = jnp.where(
          dma_sz > 0, 1, 0
      )

    kv_left_frm_new = kv_left - kv_left_frm_cache
    bkv_sz_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
    new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_frm_new)

    q_wb = jnp.maximum(0, (kv_len_start - (k_len - q_len))) // cfgs.bq_sz
    do_writeback = jnp.where((new_sz > 0) & (q_idx == q_wb), 1, 0)
    schedule.do_writeback[step, target_lane] = do_writeback
    schedule.new_sz[step, target_lane] = new_sz

    def fill_dma_kv_new(i, dma_sz, slot_start):
      dma_entry = schedule.dma_kv_new[step, target_lane, i]
      cache_pages = pl.cdiv(bkv_sz_cache, cfgs.serve.page_size)
      hbm_token_idx_base = q_end - kv_left_frm_new


      new_tok_offset = hbm_token_idx_base % cfgs.serve.page_size
      num_pages_to_fetch = jnp.where(
          new_sz > 0,
          (new_tok_offset + new_sz - 1) // cfgs.serve.page_size + 1,
          0,
      )
      fetch_val = jnp.where(i < num_pages_to_fetch, 1, 0)
      new_page_start = (
          hbm_token_idx_base - new_tok_offset
      ) + i * cfgs.serve.page_size
      fetch_vmem = (cache_pages + i) * cfgs.serve.page_size
      p_idx = jnp.minimum(
          (kv_len_start + slot_start) >> cfgs.serve.page_size_log2,
          cfgs.serve.pages_per_seq - 1,
      )
      dst_hbm_idx = jnp.minimum(
          s_idx * cfgs.serve.pages_per_seq + p_idx,
          cfgs.serve.num_page_indices - 1,
      )
      hbm_p_idx = page_indices_ref[dst_hbm_idx]
      wb_val = jnp.where(dma_sz > 0, 1, 0)

      dma_entry.fetch_hbm[...] = new_page_start
      dma_entry.fetch_vmem[...] = fetch_vmem
      dma_entry.wb_hbm[...] = hbm_p_idx
      dma_entry.wb_vmem[...] = slot_start
      dma_entry.set_flags(fetch_val, wb_val)

    if cfgs.one_new_token:
      assert cfgs.bkv_p_new == 1
      slot_start = (bkv_sz_cache // cfgs.serve.page_size) * cfgs.serve.page_size
      fill_dma_kv_new(0, new_sz, slot_start)
    else:
      iters = max(cfgs.bkv_p, cfgs.bkv_p_new)
      for i in range(iters):
        slot_start = i * cfgs.serve.page_size
        slot_end = slot_start + cfgs.serve.page_size

        dst_vmem = jnp.maximum(slot_start, bkv_sz_cache)
        end_in_slot = jnp.minimum(slot_end, bkv_sz_cache + new_sz)
        dma_sz = jnp.maximum(0, end_in_slot - dst_vmem)

        fill_dma_kv_new(i, dma_sz, slot_start)

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

    end_k_idx_causal = (
        k_len - q_len + q_idx * cfgs.bq_sz + q_sz_task - 1
    ) // cfgs.bkv_sz + 1
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
  schedule_ref.actual_steps[0] = steps  # pytype: disable=unsupported-operation

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

  # `actual_steps` reaches HBM only as part of a flush, so skipping the flush
  # leaves it holding whatever the output buffer held before. The MLA kernel
  # reads it as a grid trip count -- `cdiv(actual_steps, max_steps_ub)` -- so
  # stale contents send the step loop past the end of the schedule. Copy it
  # explicitly in the empty case.
  @pl.when(count == 0)
  def _():
    copy_actual_steps = pltpu.make_async_copy(
        schedule_ref.actual_steps.at[pl.ds(0, 1)],
        schedule_hbm_ref.actual_steps.at[pl.ds(0, 1)],
        dma_sem.at[0],
    )
    copy_actual_steps.start()
    copy_actual_steps.wait()


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
      cfgs, multiplier=cfgs.max_schedule_size_multiplier
  )

  return pl.pallas_call(
      functools.partial(rpa_metadata_schedule_kernel, cfgs=cfgs),
      out_shape=schedule_hbm,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=4,
          in_specs=[],
          out_specs=schedule_hbm.out_specs(),
          scratch_shapes=[
              schedule_shaped_dtype.scratch_shapes(),
              pltpu.SemaphoreType.DMA((1,)),
          ],
      ),
      interpret=interpret,
      name="mla_metadata_schedule",
  )(cu_q_lens, kv_lens, page_indices, distribution)