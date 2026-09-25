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
# ==============================================================================
"""Pallas BufferedRef overrides for Batched MLA DMA pipelines.

  - KVBufferedRefSeqAlongLane: Handles transposed [C_kv, K_pe] cache with dual new KV inputs.
  - BatchingQRef: combined query [Q_nope, Q_pe], one buffer and one
      DMA stream rather than two.
  - BatchingORef: Output activations (d_nope = 512).
"""

import dataclasses
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tpu_inference.kernels.mla.v3 import configs
from tpu_inference.kernels.mla.v3 import schedule

# The serving stack pins jax 0.9.2, which has not promoted BufferType out of the
# private pipeline module yet. It is the only API this kernel uses that 0.9.2
# does not expose publicly.
if hasattr(pltpu, "BufferType"):
  BufferType = pltpu.BufferType
else:  # jax < 0.11
  from jax._src.pallas.mosaic.pipeline import BufferType  # pylint: disable=g-import-not-at-top


def _tok_slice(ref, idx, off, sz, *, dim=slice(None)):
  """Slice `sz` tokens starting at `off` from a paged or batched KV ref.

  The token axis is minormost, so this is a lane slice. `dim` selects the
  latent/RoPE half on the other axis.
  """
  return ref.at[idx, dim, pl.ds(off, sz)]


def _tok_slice_2d(ref, off, sz):
  """Same, for the un-paged [kv_dim, tokens] new-KV refs."""
  return ref.at[:, pl.ds(off, sz)]


def _kv_wait_region(window_ref, cfgs, n_tokens):
  """The u32 view and slice covering `n_tokens` tokens of the KV staging buffer.

  Waiting on a DMA means describing the bytes that must have landed. The buffer
  flattens to [kv_dim/packing, tokens] and the token count slices the minor
  axis.
  """
  u32 = window_ref.bitcast(jnp.uint32)
  flat = u32.reshape((-1, cfgs.kv_vmem_lanes))
  rows = cfgs.aligned_kv_dim // cfgs.serve.packing_kv
  return flat.at[:rows, pl.ds(0, n_tokens)]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _BypassRef(pltpu.BufferedRef):
  """Helper class to safely bypass buffer_count checks during creation."""

  def __post_init__(self):
    # Pallas restricts n_buffer > 2 for output refs by default; override to allow
    # flexible pipelined buffer depths.
    pass


def _rebuild_with_cfgs(cls, standard_ref, cfgs):
  """Re-wraps a plain `pltpu.BufferedRef` as `cls`, attaching `cfgs`.

  Every subclass in this module adds exactly one field (`cfgs`) to
  `pltpu.BufferedRef`, so construction is a field-by-field copy of the standard
  ref plus that field. The copy is keyed by field *name*, so a Pallas release
  that renames or drops a `BufferedRef` field raises `TypeError` here rather
  than silently mis-wiring a buffer.

  This reaches into Pallas internals: `pltpu.BufferedRef` is not a stable public
  API. Verified against jax 0.11.x; revisit on JAX upgrades.
  """
  return cls(
      cfgs=cfgs,
      **{
          f.name: getattr(standard_ref, f.name)
          for f in dataclasses.fields(pltpu.BufferedRef)
      },
  )


# ==============================================================================
# Transposed KV Cache BufferedRef (SEQ_ALONG_LANE)
# ==============================================================================


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class KVBufferedRefSeqAlongLane(_BypassRef):
  """Handles fetching/updating KV cache using SEQ_ALONG_LANE memory layout."""

  cfgs: configs.MlaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(  # pytype: disable=signature-mismatch
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: Any,
      buffer_type: BufferType,
      buffer_count: int,
      use_lookahead: bool,
      cfgs: configs.MlaConfigs,
      **kwargs,
  ) -> "KVBufferedRefSeqAlongLane":
    assert buffer_type == BufferType.INPUT_OUTPUT

    standard_ref = _BypassRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=use_lookahead,
        **kwargs,
    )
    return _rebuild_with_cfgs(cls, standard_ref, cfgs)

  @jax.named_scope("kv_copy_in")
  def copy_in(
      self,
      src_ref: tuple[Any, Any, Any, schedule.MlaSchedule, Any],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    # src_ref: (kv_cache_hbm, new_kv_c_hbm, new_k_pe_hbm, schedule_ref, page_indices_ref)
    (
        kv_cache_hbm,
        new_kv_c_hbm,
        new_k_pe_hbm,
        schedule_ref,
        page_indices_ref,
    ) = src_ref
    del page_indices_ref

    slot = self.current_copy_in_slot
    assert self.sem_recvs is not None
    sem: Any = self.sem_recvs.at[slot]
    block_idx = jnp.maximum(grid_indices[0], 0)

    assert self.window_ref is not None
    vmem_dst_lane: Any = self.window_ref.at[slot]
    num_lanes = pltpu.get_tpu_info().num_lanes
    aligned_lkv_dim = self.cfgs.aligned_lkv_dim

    for b in range(self.cfgs.batch_size):
      # 1. Fetch cached paged tokens from 3D HBM cache
      with jax.named_scope("fetch_paged_kv_cache"):
        for i in range(self.cfgs.bkv_p_cache):
          hbm_p_idx, dma_valid = schedule_ref.get_dma_kv_cache(block_idx, b, i)
          # Compile-time constant: page i of the block lands at i * page_size.
          dst_off = i * self.cfgs.serve.page_size
          sz = jnp.where(dma_valid == 1, self.cfgs.serve.page_size, 0)
          sz = pl.multiple_of(sz, num_lanes)

          # C_kv [:aligned_lkv_dim] and K_pe [aligned_lkv_dim:] are adjacent
          # and share source and destination lane slices, so one copy is
          # identical to the two below and halves the descriptor count.
          def _start_cache(hbm_p_idx=hbm_p_idx, dst_off=dst_off, sz=sz, b=b):
            pltpu.make_async_copy(
                _tok_slice(kv_cache_hbm, hbm_p_idx, 0, sz),
                _tok_slice(vmem_dst_lane, b, dst_off, sz),
                sem,
            ).start()

          _start_cache()
      # 2. Fetch unpaged new KV tokens from HBM
      with jax.named_scope("fetch_new_kv"):
        for i in range(self.cfgs.bkv_p_new):
          dma_entry = schedule_ref.dma_kv_new[block_idx, b, i]
          src_new_off = dma_entry.fetch_hbm[...]
          dst_vmem_off = dma_entry.fetch_vmem[...]
          sz = jnp.where(dma_entry.fetch_val == 1, self.cfgs.serve.page_size, 0)
          src_new_off = pl.multiple_of(src_new_off, num_lanes)
          dst_vmem_off = pl.multiple_of(dst_vmem_off, num_lanes)
          sz = pl.multiple_of(sz, num_lanes)

          # Gating these on `fetch_val` -- only one step in three carries a
          # new token at bkv=3 over 9 pages -- was measured and is a null
          # result: bit-identical output, and the per-step deficit against v2
          # was unchanged (+11.4/+6.7/+2.5 us ungated vs +11.5/+6.5/+2.6
          # gated, at 48/24/12 steps). Zero-size DMAs are free to issue; the
          # per-step cost is elsewhere.
          def _start_new_kv(
              src_new_off=src_new_off, dst_vmem_off=dst_vmem_off, sz=sz, b=b
          ):
            # new_kv_c_hbm -> the C_kv half of the staging buffer
            with jax.named_scope("fetch_new_kv_c"):
              pltpu.make_async_copy(
                  _tok_slice_2d(new_kv_c_hbm, src_new_off, sz),
                  _tok_slice(vmem_dst_lane, b, dst_vmem_off, sz,
                             dim=slice(None, aligned_lkv_dim)),
                  sem,
              ).start()

            # new_k_pe_hbm -> the K_pe half.
            with jax.named_scope("fetch_new_k_pe"):
              pltpu.make_async_copy(
                  _tok_slice_2d(new_k_pe_hbm, src_new_off, sz),
                  _tok_slice(vmem_dst_lane, b, dst_vmem_off, sz,
                             dim=slice(aligned_lkv_dim, None)),
                  sem,
              ).start()

          _start_new_kv()

  def copy_out(
      self,
      dst_ref: tuple[jax.Ref, jax.Ref, jax.Ref, schedule.MlaSchedule, jax.Ref],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    with jax.named_scope("kv_copy_out"):
      # dst_ref: (kv_out_ref, _, _, schedule_ref, page_indices_ref)
      kv_out_ref, _, _, schedule_ref, page_indices_ref = dst_ref
      del page_indices_ref
      slot = self.current_copy_out_slot
      assert self.sem_sends is not None
      sem: Any = self.sem_sends.at[slot]
      block_idx = grid_indices[0]

      assert self.window_ref is not None
      vmem_src_lane: Any = self.window_ref.at[slot]
      num_lanes = pltpu.get_tpu_info().num_lanes
      aligned_lkv_dim = self.cfgs.aligned_lkv_dim

      for b in range(self.cfgs.batch_size):
        do_writeback = schedule_ref.do_writeback[block_idx, b] == 1
        for i in range(self.cfgs.bkv_p_new):
          dma_entry = schedule_ref.dma_kv_new[block_idx, b, i]
          hbm_p_idx = dma_entry.wb_hbm[...]
          src_vmem_off = dma_entry.wb_vmem[...]
          dma_valid = dma_entry.wb_val
          sz = jnp.where(do_writeback, dma_valid * self.cfgs.serve.page_size, 0)
          src_vmem_off = pl.multiple_of(src_vmem_off, num_lanes)
          sz = pl.multiple_of(sz, num_lanes)

          # Write back concatenated into 3D kv_out_ref. Same adjacency argument
          # as `copy_in`: one copy covers both halves.
          def _start_wb(
              hbm_p_idx=hbm_p_idx, src_vmem_off=src_vmem_off, sz=sz, b=b
          ):
            pltpu.make_async_copy(
                _tok_slice(vmem_src_lane, b, src_vmem_off, sz),
                _tok_slice(kv_out_ref, hbm_p_idx, 0, sz),
                sem,
            ).start()

          _start_wb()

  def wait_in(
      self,
      src_ref: tuple[jax.Ref, jax.Ref, jax.Ref, schedule.MlaSchedule, jax.Ref],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    with jax.named_scope("wait_kv_copy_in"):
      _, _, _, schedule_ref, _ = src_ref
      slot = self.current_wait_in_slot
      assert self.sem_recvs is not None
      sem: Any = self.sem_recvs.at[slot]
      block_idx = grid_indices[0]

      # The schedule already summed this when it built the descriptors. This
      # used to re-derive it from batch_size * (bkv_p_cache + bkv_p_new) SMEM
      # reads on every grid step -- 16 at batch=4, bkv=3 -- which is per-step
      # work v2 does not do, and the deficit against v2 is per-grid-step.
      kv_in_tokens = pl.multiple_of(
          jnp.asarray(schedule_ref.total_wait_kv_in[block_idx]), 128
      )

      assert self.window_ref is not None
      vmem_dst: Any = self.window_ref.at[slot]
      region = _kv_wait_region(vmem_dst, self.cfgs, kv_in_tokens)
      pltpu.make_async_copy(region, region, sem).wait()

  def wait_out(
      self,
      dst_ref: tuple[jax.Ref, jax.Ref, jax.Ref, schedule.MlaSchedule, jax.Ref],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    with jax.named_scope("wait_kv_copy_out"):
      _, _, _, schedule_ref, _ = dst_ref
      slot = self.current_wait_out_slot
      assert self.sem_sends is not None
      sem: Any = self.sem_sends.at[slot]
      block_idx = grid_indices[0]

      # As in `wait_in`: precomputed by the schedule.
      kv_out_tokens = pl.multiple_of(
          jnp.asarray(schedule_ref.total_wait_kv_out[block_idx]), 128
      )

      assert self.window_ref is not None
      vmem_src: Any = self.window_ref.at[slot]
      region = _kv_wait_region(vmem_src, self.cfgs, kv_out_tokens)
      pltpu.make_async_copy(region, region, sem).wait()


# ==============================================================================
# Dedicated Query BufferedRef (BatchingQRef)
# ==============================================================================


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class BatchingQRef(pltpu.BufferedRef):
  """Handles fetching Query block [Q_nope, Q_pe]."""

  cfgs: configs.MlaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(  # pytype: disable=signature-mismatch
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: Any,
      buffer_type: BufferType,
      buffer_count: int,
      use_lookahead: bool,
      cfgs: configs.MlaConfigs,
      **kwargs,
  ) -> "BatchingQRef":
    # `BufferType` rather than `pltpu.BufferType`: jax 0.9.2, which the serving
    # stack pins, has not promoted it to the public namespace yet. See the
    # shim at the top of this module.
    assert buffer_type == BufferType.INPUT

    standard_ref = pltpu.BufferedRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=use_lookahead,
        **kwargs,
    )
    return _rebuild_with_cfgs(cls, standard_ref, cfgs)

  @jax.named_scope("q_copy_in")
  def copy_in(
      self,
      src_ref: tuple[jax.Ref, jax.Ref, schedule.MlaSchedule],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    # src_ref: (q_nope_hbm, q_pe_hbm, schedule_ref)
    q_nope_hbm, q_pe_hbm, schedule_ref = src_ref
    slot = self.current_copy_in_slot
    assert self.sem_recvs is not None
    sem: Any = self.sem_recvs.at[slot]
    assert self.window_ref is not None
    vmem_dst: Any = self.window_ref.at[slot]
    block_idx = grid_indices[0]

    aligned_lkv_dim = self.cfgs.aligned_lkv_dim

    for b in range(self.cfgs.batch_size):
      q_src, q_sz = schedule_ref.get_dma_q(block_idx, b)

      # Copy Q_nope
      pltpu.make_async_copy(
          q_nope_hbm.at[pl.ds(q_src, q_sz), ...],
          vmem_dst.at[b, pl.ds(0, q_sz), :, :aligned_lkv_dim],
          sem,
      ).start()

      # Copy Q_pe
      pltpu.make_async_copy(
          q_pe_hbm.at[pl.ds(q_src, q_sz), ...],
          vmem_dst.at[b, pl.ds(0, q_sz), :, aligned_lkv_dim:],
          sem,
      ).start()

  def wait_in(
      self,
      src_ref: tuple[jax.Ref, jax.Ref, schedule.MlaSchedule],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    _, _, schedule_ref = src_ref
    slot = self.current_wait_in_slot
    assert self.sem_recvs is not None
    sem: Any = self.sem_recvs.at[slot]
    block_idx = grid_indices[0]

    # Summed by the schedule when it built the q descriptors; this loop
    # re-read `batch_size` of them on every grid step.
    q_in_tokens = schedule_ref.total_wait_q_in[block_idx]

    itemsize = jnp.dtype(self.cfgs.serve.dtype_q).itemsize
    minor = self.cfgs.aligned_q_dim
    dma_chunk_size = minor * 4
    q_bytes_per_token = self.cfgs.q_bytes_per_token

    rows_per_token = q_bytes_per_token // dma_chunk_size
    assert rows_per_token % 8 == 0, (
        f"{rows_per_token=} must be sublane-aligned for the wait slice"
    )
    wait_lanes = q_in_tokens * rows_per_token

    assert self.window_ref is not None
    vmem_dst: Any = self.window_ref.at[slot]
    vmem_u32 = vmem_dst.bitcast(jnp.uint32)
    flat_vmem = vmem_u32.reshape((-1, minor))
    pltpu.make_async_copy(
        flat_vmem.at[pl.ds(0, wait_lanes), :],
        flat_vmem.at[pl.ds(0, wait_lanes), :],
        sem,
    ).wait()


# ==============================================================================
# Output Activation BufferedRef (BatchingORef)
# ==============================================================================


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class BatchingORef(pltpu.BufferedRef):
  """Handles storing final attention output activations (width d_nope = 512)."""

  cfgs: configs.MlaConfigs = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(  # pytype: disable=signature-mismatch
      cls,
      spec: pl.BlockSpec,
      dtype_or_type: Any,
      buffer_type: BufferType,
      buffer_count: int,
      use_lookahead: bool,
      cfgs: configs.MlaConfigs,
      **kwargs,
  ) -> "BatchingORef":
    assert buffer_type == BufferType.OUTPUT

    standard_ref = pltpu.BufferedRef.create(
        spec=spec,
        dtype_or_type=dtype_or_type,
        buffer_type=buffer_type,
        buffer_count=buffer_count,
        grid_rank=1,
        use_lookahead=use_lookahead,
        **kwargs,
    )
    return _rebuild_with_cfgs(cls, standard_ref, cfgs)

  def copy_out(
      self,
      dst_ref: tuple[jax.Ref, schedule.MlaSchedule],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    # dst_ref: (o_hbm, schedule_ref)
    o_hbm, schedule_ref = dst_ref
    slot = self.current_copy_out_slot
    assert self.sem_sends is not None
    sem: Any = self.sem_sends.at[slot]
    assert self.window_ref is not None
    vmem_src: Any = self.window_ref.at[slot]
    block_idx = grid_indices[0]

    for b in range(self.cfgs.batch_size):
      is_last_k = schedule_ref.is_last_k[block_idx, b] == 1
      q_src, q_sz = schedule_ref.get_dma_q(block_idx, b)
      q_sz = jnp.where(is_last_k, q_sz, 0)

      def _start_o(q_src=q_src, q_sz=q_sz, b=b):
        pltpu.make_async_copy(
            vmem_src.at[b, pl.ds(0, q_sz), ...],
            o_hbm.at[pl.ds(q_src, q_sz), ...],
            sem,
        ).start()

      # Output is emitted only at a sequence's last k-block; the rest issue a
      # zero-length copy. `wait_out` counts bytes, so this is invisible to it.
      _start_o()

  def wait_out(
      self,
      dst_ref: tuple[jax.Ref, schedule.MlaSchedule],
      grid_indices: tuple[int | jax.Array, ...],
  ):
    # dst_ref: (o_hbm, schedule_ref)
    _, schedule_ref = dst_ref
    slot = self.current_wait_out_slot
    assert self.sem_sends is not None
    sem: Any = self.sem_sends.at[slot]
    block_idx = grid_indices[0]
    # `_compute_waits` emits this in rows of the same `minor`-wide u32 view
    # the slice below uses, so it needs no rescaling here. Mosaic cannot see
    # that a value read out of SMEM is sublane-aligned, hence the hint; the
    # schedule asserts the property where the operands are Python ints.
    minor = self.cfgs.aligned_lkv_dim
    itemsize = jnp.dtype(self.cfgs.serve.dtype_out).itemsize
    o_bytes_per_token = (
        self.cfgs.aligned_num_q_heads * self.cfgs.aligned_lkv_dim * itemsize
    )
    rows_per_token = o_bytes_per_token // (minor * 4)
    assert rows_per_token % 8 == 0
    # As above: precomputed by the schedule.
    o_tokens = schedule_ref.total_wait_o_out[block_idx]
    wait_lanes = o_tokens * rows_per_token
    assert self.window_ref is not None
    vmem_src: Any = self.window_ref.at[slot]
    vmem_u32 = vmem_src.bitcast(jnp.uint32)
    flat_src = vmem_u32.reshape((-1, minor))
    pltpu.make_async_copy(
        flat_src.at[pl.ds(0, wait_lanes), :],
        flat_src.at[pl.ds(0, wait_lanes), :],
        sem,
    ).wait()


