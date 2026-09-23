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
"""TPU-Friendly MLA Ragged Paged Attention kernel v3 (Transposed KV Cache)."""

import dataclasses
import functools
from typing import Any

from absl import logging
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tpu_inference import envs
from tpu_inference.kernels.mla.v2.transpose import xpose_pipeline
from tpu_inference.kernels.mla.v3 import bref_override
from tpu_inference.kernels.mla.v3 import configs
from tpu_inference.kernels.mla.v3 import flash_attention
from tpu_inference.kernels.mla.v3 import schedule
from tpu_inference.kernels.mla.v3 import stitch_utils
from tpu_inference.kernels.mla.v3 import utils

MlaCase = configs.MlaCase

# Same tile v2 uses for the output-side transpose.
_XPOSE_N_TILE_SIZE = envs.MLA_XPOSE_N_TILE_SIZE


# ==============================================================================
# Input Validation & Preparation (Transposed KV Cache Only)
# ==============================================================================


def static_validate_inputs(
    ql_nope: jax.Array,  # [actual_num_q_heads, max_num_tokens, actual_lkv_dim]
    q_pe: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [max_num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [max_num_tokens, actual_r_dim]
    cache_kv: jax.Array,  # [total_num_pages, aligned_kv_dim // kv_packing, kv_packing, page_size]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    cfgs: configs.MlaConfigs,
) -> None:
  """Statically validates input shapes, dtypes, and 4D transposed layout constraints."""
  if len(ql_nope.shape) != 3:
    raise ValueError(f"Expected 3D array for {ql_nope.shape=}")
  if len(q_pe.shape) != 3:
    raise ValueError(f"Expected 3D array for {q_pe.shape=}")
  if len(new_kv_c.shape) != 2:
    raise ValueError(f"Expected 2D array for {new_kv_c.shape=}")
  if len(new_k_pe.shape) != 2:
    raise ValueError(f"Expected 2D array for {new_k_pe.shape=}")

  if ql_nope.shape[0] != q_pe.shape[1]:
    raise ValueError(
        f"Expected ql_nope num_heads {ql_nope.shape[0]=} to equal"
        f" q_pe num_heads {q_pe.shape[1]=}"
    )
  if ql_nope.shape[1] != q_pe.shape[0]:
    raise ValueError(
        f"Expected ql_nope num_tokens {ql_nope.shape[1]=} to equal"
        f" q_pe num_tokens {q_pe.shape[0]=}"
    )
  if ql_nope.shape[1] != new_kv_c.shape[0]:
    raise ValueError(
        f"Expected {ql_nope.shape[1]=} to be equal to {new_kv_c.shape[0]=}"
    )
  if new_kv_c.shape[0] != new_k_pe.shape[0]:
    raise ValueError(
        f"Expected {new_kv_c.shape[0]=} to be equal to {new_k_pe.shape[0]=}"
    )
  if ql_nope.shape[2] != new_kv_c.shape[1]:
    raise ValueError(
        f"Expected {ql_nope.shape[2]=} to be equal to {new_kv_c.shape[1]=}"
    )
  if q_pe.shape[2] != new_k_pe.shape[1]:
    raise ValueError(
        f"Expected {q_pe.shape[2]=} to be equal to {new_k_pe.shape[1]=}"
    )

  actual_lkv_dim = ql_nope.shape[-1]
  actual_r_dim = q_pe.shape[-1]
  # Must mirror `MlaConfigs.kv_dim_align`: each part is padded to 128 lanes.
  align = cfgs.kv_dim_align
  lkv_dim = utils.align_to(actual_lkv_dim, align)
  r_dim = utils.align_to(actual_r_dim, align)

  if cache_kv.ndim != 3:
    raise ValueError(
        "Expected 3D array [pages, aligned_kv_dim, page_size] for cache_kv,"
        f" got {cache_kv.shape=}"
    )

  total_num_pages, kv_dim, page_size = cache_kv.shape

  if page_size % 128 != 0:
    raise ValueError(f"Expected {page_size=} to be a multiple of 128.")

  if kv_dim != cfgs.aligned_kv_dim:
    raise ValueError(
        f"cache_kv {kv_dim=} does not match {cfgs.aligned_kv_dim=}"
        f" (= aligned_lkv_dim {cfgs.aligned_lkv_dim} + aligned_r_dim"
        f" {cfgs.aligned_r_dim})"
    )

  if not (cache_kv.dtype == new_kv_c.dtype):
    raise ValueError(
        f"Expected {cache_kv.dtype=} to be equal to {new_kv_c.dtype=}."
    )
  if not (cache_kv.dtype == new_k_pe.dtype):
    raise ValueError(
        f"Expected {cache_kv.dtype=} to be equal to {new_k_pe.dtype=}."
    )

  if not jnp.issubdtype(cache_kv.dtype, jnp.floating):
    raise ValueError(f"Expected {cache_kv.dtype=} to be a floating point.")

  if not (
      jnp.int32
      == kv_lens.dtype
      == page_indices.dtype
      == cu_q_lens.dtype
      == distribution.dtype
  ):
    raise ValueError(
        f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
        f" {cu_q_lens.dtype=}, {distribution.dtype=}"
    )

  if not (
      len(kv_lens.shape) == len(page_indices.shape) == len(cu_q_lens.shape) == 1
  ):
    raise ValueError(
        f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
        f" {cu_q_lens.shape=}"
    )

  max_num_seqs = kv_lens.shape[0]
  num_page_indices = page_indices.shape[0]
  if num_page_indices % max_num_seqs != 0:
    raise ValueError(
        f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
    )
  if cu_q_lens.shape != (max_num_seqs + 1,):
    raise ValueError(
        f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)."
    )
  if distribution.shape != (3,):
    raise ValueError(f"Expected {distribution.shape=} to be (3,).")

  # `page_size_log2` implements `//` as a shift, which is only equivalent for
  # powers of two. A multiple of 128 is not enough (e.g. 384).
  if page_size & (page_size - 1) != 0:
    raise ValueError(f"Expected {page_size=} to be a power of two.")

  if cfgs.serve.page_size != page_size:
    raise ValueError(
        f"Config {cfgs.serve.page_size=} disagrees with the cache_kv page"
        f" dimension {page_size=}."
    )

  # `bkv_p` is `cdiv(bkv_sz, page_size)`, so a non-multiple would silently round
  # the block up and make every DMA offset computed from `bkv_p * page_size`
  # disagree with the `bkv_sz` the softmax masks against.
  if cfgs.bkv_sz % page_size != 0:
    raise ValueError(
        f"Expected {cfgs.bkv_sz=} to be a multiple of {page_size=}."
    )

  if cfgs.model.sliding_window is not None and cfgs.model.sliding_window <= 0:
    raise ValueError(f"{cfgs.model.sliding_window=} must be positive.")
  if cfgs.model.soft_cap is not None and cfgs.model.soft_cap == 0.0:
    raise ValueError(f"{cfgs.model.soft_cap=} must not be 0.0.")
  if cfgs.vmem_limit_bytes <= 0:
    raise ValueError(f"{cfgs.vmem_limit_bytes=} must be positive.")


def prepare_q_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
    head_align: int = 128,
) -> jax.Array:
  """Pads q to [max_num_tokens, num_q_heads, head_dim].

  Kept 3D. The packed `words x packing x 128` form this used to produce cost
  a T(32,128) <-> T(4,128) relayout at the HBM boundary that the kernel then
  undid, worth 18% on decode, so `mla_body` now consumes the 3D layout
  unconditionally.
  """
  # `head_align` must match `MlaConfigs.kv_dim_align` for q_pe: the QK-PE dot
  # contracts q_pe against k_pe over this dimension, so the two must agree.
  _, actual_num_q_heads, actual_head_dim = q.shape
  packing_q = utils.get_dtype_packing(q.dtype)
  num_q_heads = utils.align_to(actual_num_q_heads, packing_q)
  head_dim = utils.align_to(actual_head_dim, head_align)
  return jnp.pad(
      q,
      (
          (0, 0),
          (0, num_q_heads - actual_num_q_heads),
          (0, head_dim - actual_head_dim),
      ),
      constant_values=0,
  )


def _physical_transpose(x: jax.Array, *, n_tile: int, m_tile: int) -> jax.Array:
  """(N, T, D) <-> (T, N, D) via v2's pipelined transpose kernel.

  A plain `jnp.transpose` feeding a Pallas call cannot be folded into the
  producing fusion -- Pallas fixes its operand layouts -- so XLA materializes a
  layout-changing copy of the whole tensor, once per attention layer on both the
  q_nope and the output side. `xpose_pipeline` instead tiles and double-buffers
  the movement, which is why v2 treats `jnp.transpose` as the fallback it warns
  about (v2/kernel.py:1371).

  Falls back on ValueError for the same reason v2 does: the kernel rejects
  shapes it cannot tile.
  """
  try:
    return xpose_pipeline(x, transpose_axes=(1, 0, 2), n_tile=n_tile,
                          m_tile=m_tile)[0]
  except ValueError as e:
    logging.warning(
        "xpose_pipeline failed for shape=%s dtype=%s: %s. Falling back to"
        " jnp.transpose -- this materializes a full copy per layer.",
        x.shape, x.dtype, e)
    return jnp.transpose(x, (1, 0, 2))


def prepare_q_nope_inputs(
    q: jax.Array,  # [actual_num_q_heads, max_num_tokens, actual_head_dim]
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
  """Pads and physically transposes q_nope to token-major layout.

  Returns: [max_num_tokens, num_q_heads, head_dim], kept 3D for the same
  reason as `prepare_q_inputs`.
  """
  del vmem_limit_bytes
  actual_num_q_heads, actual_max_num_tokens, actual_head_dim = q.shape
  packing_q = utils.get_dtype_packing(q.dtype)
  num_q_heads = utils.align_to(actual_num_q_heads, packing_q)
  head_dim = utils.align_to(actual_head_dim, 128)

  # Token alignment drives how many rows `xpose_pipeline` moves, and the
  # transpose runs once per layer per step. Rounding to packing_q*8 = 32 costs
  # 2x at low concurrency: the framework floor is 16 tokens per DP rank
  # (tpu_runner.py:512, max(16, next_power_of_2(dp_size * kv_packing)) / dp),
  # so 16 real tokens became 32 rows while v2 transposed 16.
  #
  # The token axis is untiled in the transposed result (heads carries the
  # sublane tile there), and `xpose_pipeline` clamps m_tile to the axis size,
  # so 32 is not required -- v2 feeds it an unpadded 16 and that is the
  # `..._shape_128x16x512_..._m_tile_16` call visible in its HLO. Align to the
  # sublane tile only; the padding decays to nothing once the real token count
  # is already a multiple of 8.
  token_align = 8
  max_num_tokens = utils.align_to(actual_max_num_tokens, token_align)
  head_pad = (0, num_q_heads - actual_num_q_heads)
  token_pad = (0, max_num_tokens - actual_max_num_tokens)
  dim_pad = (0, head_dim - actual_head_dim)
  q = jnp.pad(
      q,
      (head_pad, token_pad, dim_pad),
      constant_values=0,
  )
  return _physical_transpose(q, n_tile=128, m_tile=32)


def prepare_kv_inputs_for_transposed_kv_cache(
    kv: jax.Array,
    page_size: int = 128,
    head_align: int = 128,
) -> jax.Array:
  """Pads new KV inputs to match the cache layout.

  The cache wants [aligned_head_dim, max_num_tokens], so the incoming
  [tokens, head_dim] is transposed.
  """
  max_num_tokens, actual_head_dim = kv.shape

  pad_multiple = max(128, page_size)
  if max_num_tokens % pad_multiple != 0:
    pad = pad_multiple - (max_num_tokens % pad_multiple)
    kv = jnp.pad(kv, ((0, pad), (0, 0)), constant_values=0)

  aligned_head_dim = utils.align_to(actual_head_dim, head_align)
  if aligned_head_dim != actual_head_dim:
    pad = aligned_head_dim - actual_head_dim
    kv = jnp.pad(kv, ((0, 0), (0, pad)), constant_values=0)

  return jnp.transpose(kv, (1, 0))


def prepare_outputs(
    out: jax.Array,  # [max_num_tokens, o_words, packing_q, 128]
    actual_num_q_heads: int,
    actual_max_num_tokens: int,
    actual_head_dim: int,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
  """Physically transposes output activations back to head-major layout."""
  del vmem_limit_bytes
  packing_q = utils.get_dtype_packing(out.dtype)
  num_q_heads = utils.align_to(actual_num_q_heads, packing_q)
  head_dim = utils.align_to(actual_head_dim, 128)
  out = out.reshape((out.shape[0], num_q_heads, head_dim))
  out = _physical_transpose(out, n_tile=_XPOSE_N_TILE_SIZE, m_tile=64)
  return out[:actual_num_q_heads, :actual_max_num_tokens, :actual_head_dim]


# ==============================================================================
# Pallas Allocation & Compute Pipeline
# ==============================================================================


def create_allocs(
    cache_kv_hbm_ref: jax.Ref,
    ql_nope_hbm_ref: jax.Ref,
    q_pe_hbm_ref: jax.Ref,
    o_hbm_ref: jax.Ref,
    cfgs: configs.MlaConfigs,
):
  """Instantiates BufferedRef overrides for MLA query, KV cache, and output."""
  kv_cache_spec = pl.BlockSpec(
      block_shape=cfgs.kv_vmem_shape,
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i,),
      pipeline_mode=pl.Buffered(buffer_count=cfgs.n_buffer, use_lookahead=True),
  )
  q_nope_spec = pl.BlockSpec(
      block_shape=cfgs.q_nope_vmem_shape,
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i,),
      pipeline_mode=pl.Buffered(buffer_count=cfgs.n_buffer, use_lookahead=True),
  )
  q_pe_spec = pl.BlockSpec(
      block_shape=cfgs.q_pe_vmem_shape,
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i,),
      pipeline_mode=pl.Buffered(buffer_count=cfgs.n_buffer, use_lookahead=True),
  )
  o_spec = pl.BlockSpec(
      block_shape=cfgs.o_vmem_shape,
      memory_space=pltpu.VMEM,
      index_map=lambda i: (i,),
      pipeline_mode=pl.Buffered(buffer_count=2, use_lookahead=False),
  )

  kv_cache_alloc = bref_override.KVBufferedRefSeqAlongLane.input_output(
      spec=kv_cache_spec,
      dtype_or_type=cache_kv_hbm_ref,
      buffer_count=cfgs.n_buffer,
      use_lookahead=True,
      cfgs=cfgs,
  )
  # Each buffered ref takes its element type from the HBM ref it stages, not
  # from the output ref. `ql_nope_hbm_ref` happens to alias `o_hbm_ref` (see
  # `input_output_aliases` below), but `q_pe_hbm_ref` does not, so sourcing its
  # dtype from the output would allocate the wrong-width VMEM buffer whenever
  # `dtype_out != dtype_q`.
  q_nope_alloc = bref_override.BatchingQNopeRef.input(
      spec=q_nope_spec,
      dtype_or_type=ql_nope_hbm_ref,
      buffer_count=cfgs.n_buffer,
      use_lookahead=True,
      cfgs=cfgs,
  )
  q_pe_alloc = bref_override.BatchingQPeRef.input(
      spec=q_pe_spec,
      dtype_or_type=q_pe_hbm_ref,
      buffer_count=cfgs.n_buffer,
      use_lookahead=True,
      cfgs=cfgs,
  )
  o_alloc = bref_override.BatchingORef.output(
      spec=o_spec,
      dtype_or_type=o_hbm_ref,
      buffer_count=2,
      use_lookahead=False,
      cfgs=cfgs,
  )

  return q_nope_alloc, q_pe_alloc, kv_cache_alloc, o_alloc


def calculate_and_store_out(
    step_idx: jax.Array,
    schedule_ref: schedule.MlaSchedule,
    acc: jax.Array,
    l_val: jax.Array,
    o_vref: jax.Ref,
    *,
    cfgs: configs.MlaConfigs,
):
  """Normalizes accumulated attention output by denominator l and stores to o_vref."""

  def _accum(b_idx: int | jax.Array, batch_acc: jax.Array, batch_l: jax.Array):
    exact_div = (
        cfgs.serve.dtype_out == jnp.float32
        or cfgs.serve.dtype_out == batch_l.dtype == jnp.bfloat16
    )
    batch_l = utils.broadcast_minor(batch_l, batch_acc.shape)
    if exact_div:
      result = lax.div(batch_acc, batch_l)
    else:
      result = batch_acc * pl.reciprocal(batch_l, approx=True)
    out = result.astype(cfgs.serve.dtype_out)
    out = out.reshape(
        cfgs.block.bq_sz, cfgs.aligned_num_q_heads, cfgs.aligned_lkv_dim
    )
    o_vref[b_idx, ...] = out

  if cfgs.fuse_accum:
    for b in range(cfgs.batch_size):
      _accum(b, acc[b], l_val[b])
  else:
    for b in range(cfgs.batch_size):
      is_last_k = schedule_ref.is_last_k[step_idx, b] == 1
      acc_val = acc[b]
      l_v = l_val[b]
      accum_named_call = jax.named_call(_accum, name=f"accum_{b}")
      jax.lax.cond(
          is_last_k,
          accum_named_call,
          lambda *_: None,
          b,
          acc_val,
          l_v,
      )


def mla_body(
    q_nope_vref: Any,
    q_pe_vref: Any,
    kv_in_vref: Any,
    o_vref: Any,
    schedule_ref: schedule.MlaSchedule,
    m_scratch_ref: Any,
    l_scratch_ref: Any,
    acc_scratch_ref: Any,
    *,
    cu_q_lens_ref: Any,
    kv_lens_ref: Any,
    cfgs: configs.MlaConfigs,
):
  """Inner step execution body of the MLA Pallas pipeline."""
  step = pl.program_id(0)

  with jax.named_scope("doing_math"):
    processed_q_len = []
    processed_kv_len = []
    bkv_sz_frm_cache_list = []
    new_kv_len_start_list = []
    new_sz_list = []
    for b_idx in range(cfgs.batch_size):
      s_idx = schedule_ref.s_idx[step, b_idx]
      is_valid = s_idx != -1
      q_idx = schedule_ref.q_idx[step, b_idx]
      k_idx = schedule_ref.k_idx[step, b_idx]
      k_id = jnp.where(is_valid, k_idx * cfgs.bkv_sz, 0)
      kv_len = jnp.where(is_valid, kv_lens_ref[s_idx], 0)
      q_start = jnp.where(is_valid, cu_q_lens_ref[s_idx], 0)
      q_end = jnp.where(is_valid, cu_q_lens_ref[s_idx + 1], 0)
      q_len = q_end - q_start
      offset = kv_len - q_len

      processed_q_len.append((q_idx * cfgs.bq_sz + offset))
      processed_kv_len.append(k_id)

      # Stitching metadata
      kv_left = jnp.maximum(kv_len - k_id, 0)
      kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
      kv_left_frm_new = jnp.maximum(kv_left - kv_left_frm_cache, 0)

      bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
      new_kv_len_start = q_end - kv_left_frm_new

      bkv_sz_frm_cache_list.append(bkv_sz_frm_cache)
      new_kv_len_start_list.append(new_kv_len_start)
      # Mirrors `schedule.k_loop`'s `new_sz`: how many unpaged new tokens land in
      # this block. Zero for every step except the one holding the sequence's
      # new token, which is what `gate_stitch` keys on.
      new_sz_list.append(
          jnp.minimum(cfgs.bkv_sz - bkv_sz_frm_cache, kv_left_frm_new)
      )

  # Skip the merge -- strided loads, roll, strided stores -- on lanes with no
  # new tokens. Only one step per sequence has any, so ~2/3 of lane-steps
  # elide. Each lane owns a disjoint slot, so there is no cross-lane hazard in
  # skipping the store. -2.7% on decode.
  for b_idx in range(cfgs.batch_size):

    @pl.when(new_sz_list[b_idx] > 0)
    def _stitch_and_store(b_idx=b_idx):
      with jax.named_scope("stitch_and_store"):
        stitch_utils.store_new_kv_lane(
            kv_in_vref,
            b_idx,
            stitch_utils.stitch_new_kv_lane(
                kv_in_vref,
                b_idx,
                bkv_sz_frm_cache_list[b_idx],
                new_kv_len_start_list[b_idx],
                cfgs=cfgs,
            ),
            cfgs=cfgs,
        )

  with jax.named_scope("load_q_pe"):
    q_nope = q_nope_vref.reshape(cfgs.batch_size, -1, cfgs.aligned_lkv_dim)
    q_pe = q_pe_vref.reshape(cfgs.batch_size, -1, cfgs.aligned_r_dim)

  with jax.named_scope("load_kv"):
    # `.at[]` takes a view rather than loading. `chunked_flash_attention`
    # accepts a Ref and loads at the point of use, so the KV tile is not live
    # across everything in between -- which is what costs registers on DECODE.
    c_kv = kv_in_vref.at[:, : cfgs.aligned_lkv_dim, : cfgs.bkv_sz]
    k_pe = kv_in_vref.at[:, cfgs.aligned_lkv_dim :, : cfgs.bkv_sz]

  is_last_k_list = [
      schedule_ref.is_last_k[step, b] == 1 for b in range(cfgs.batch_size)
  ]

  flash_attention.chunked_flash_attention(
      q_nope=q_nope,
      q_pe=q_pe,
      k_nope=c_kv,
      k_pe=k_pe,
      m_scratch_ref=m_scratch_ref,
      l_scratch_ref=l_scratch_ref,
      acc_scratch_ref=acc_scratch_ref,
      o_vref=o_vref,
      is_last_k=is_last_k_list,
      processed_q_len=processed_q_len,
      processed_kv_len=processed_kv_len,
      cfgs=cfgs,
  )


def get_kernel_name(cfgs: configs.MlaConfigs) -> str:
  serve = cfgs.serve
  name = f"V3-MLA{cfgs.mode.symbol}-{serve.kv_layout.symbol}-p{serve.page_size}"
  name += f"-b{cfgs.batch_size}-q{cfgs.bq_sz}-k{cfgs.bkv_sz}"
  if cfgs.model.sliding_window:
    name += f"-sw{cfgs.model.sliding_window}"
  return name


def get_kernel_metadata(
    cfgs: configs.MlaConfigs,
) -> dict[str, str | int | float]:
  cfgs_dict = dataclasses.asdict(cfgs)
  ret = {}
  for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
    key = jax.tree_util.keystr(path, simple=True, separator=".")
    if not isinstance(val, (str, int, float)):
      val = str(val)
    ret[key] = val
  return ret


def _mla_ragged_paged_attention_kernel(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    schedule_hbm: schedule.MlaSchedule,
    ql_nope_hbm: jax.Array,
    q_pe_hbm: jax.Array,
    new_kv_c_hbm: jax.Array,
    new_k_pe_hbm: jax.Array,
    cache_kv_hbm: jax.Array,
    *,
    cfgs: configs.MlaConfigs,
) -> tuple[jax.Array, jax.Array]:
  """Executes the Pallas MLA attention pipeline with HBM schedule data."""

  def ragged_paged_attention_pipeline(
      # Scalar prefetch.
      cu_q_lens_ref: jax.Ref,
      kv_lens_ref: jax.Ref,
      page_indices_ref: jax.Ref,
      # Inputs.
      schedule_hbm_ref: schedule.MlaSchedule,
      ql_nope_hbm_ref: jax.Ref,
      q_pe_hbm_ref: jax.Ref,
      new_kv_c_hbm_ref: jax.Ref,
      new_k_pe_hbm_ref: jax.Ref,
      cache_kv_hbm_ref: jax.Ref,
      # Outputs.
      o_hbm_ref: jax.Ref,
      o_kv_cache_hbm_ref: jax.Ref,
  ):
    del o_kv_cache_hbm_ref

    q_nope_alloc, q_pe_alloc, kv_cache_alloc, o_alloc = create_allocs(
        cache_kv_hbm_ref, ql_nope_hbm_ref, q_pe_hbm_ref, o_hbm_ref, cfgs
    )

    actual_steps = schedule_hbm_ref.actual_steps[0]
    num_safe_step_iterations = pl.cdiv(actual_steps, cfgs.max_steps_ub)

    @pl.with_scoped(
        final_allocs=(q_nope_alloc, q_pe_alloc, kv_cache_alloc, o_alloc),
        schedule_ref=schedule.MlaSchedule.create_shape_dtype(
            cfgs
        ).scratch_shapes(),
        dma_sem=pltpu.SemaphoreType.DMA((1,)),
        scratches=(
            pltpu.VMEM(
                cfgs.lm_scratch_shape,
                dtype=jnp.float32,
            ),  # m
            pltpu.VMEM(
                cfgs.lm_scratch_shape,
                dtype=jnp.float32,
            ),  # l
            pltpu.VMEM(
                cfgs.acc_scratch_shape,
                dtype=jnp.float32,
            ),  # acc
        ),
    )
    def _run(final_allocs, schedule_ref, dma_sem, scratches):
      scratches[0][...] = jnp.full_like(scratches[0], -jnp.inf)
      scratches[1][...] = jnp.zeros_like(scratches[1])
      scratches[2][...] = jnp.zeros_like(scratches[2])

      kv_alloc = final_allocs[2]
      assert kv_alloc.window_ref is not None
      kv_alloc.window_ref[...] = jnp.zeros_like(kv_alloc.window_ref)

      def execute_schedule_chunk(start_step, num_steps):
        aligned_start_step = (start_step // 128) * 128
        prefix_steps = start_step - aligned_start_step

        flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)
        flat_smem = jax.tree_util.tree_leaves(schedule_ref)
        dma_list = []
        for h, s in zip(flat_hbm, flat_smem):
          if jax.typeof(h).memory_space == pltpu.HBM:
            element_size = s.shape[0] // cfgs.max_steps_ub
            read_size = element_size * (num_steps + prefix_steps)
            read_size = utils.align_to(read_size, 1024)
            read_size = jnp.minimum(read_size, s.shape[0])

            src_off = element_size * aligned_start_step
            src_off = pl.multiple_of(src_off, 128)

            copy = pltpu.make_async_copy(
                h.at[pl.ds(src_off, read_size)],
                s.at[pl.ds(0, read_size)],
                dma_sem.at[0],
            )
            copy.start()
            dma_list.append(copy)
        jax.tree.map(lambda x: x.wait(), dma_list)

        pipeline_func = pltpu.emit_pipeline(
            body=functools.partial(
                mla_body,
                cfgs=cfgs,
                cu_q_lens_ref=cu_q_lens_ref,
                kv_lens_ref=kv_lens_ref,
            ),
            grid=(num_steps + prefix_steps,),
            in_specs=(
                q_nope_alloc.spec,
                q_pe_alloc.spec,
                kv_cache_alloc.spec,
            ),
            out_specs=(o_alloc.spec,),
        )
        pipeline_func(
            (ql_nope_hbm_ref, schedule_ref),
            (q_pe_hbm_ref, schedule_ref),
            (
                cache_kv_hbm_ref,
                new_kv_c_hbm_ref,
                new_k_pe_hbm_ref,
                schedule_ref,
                page_indices_ref,
            ),
            (o_hbm_ref, schedule_ref),
            scratches=(schedule_ref,) + scratches,
            allocations=final_allocs,
        )

      @pl.loop(0, num_safe_step_iterations)
      def loop_body(step_idx):
        start = step_idx * cfgs.max_steps_ub
        rem = actual_steps % cfgs.max_steps_ub
        last_step_size = jnp.where(rem == 0, cfgs.max_steps_ub, rem)
        is_last_step = step_idx == num_safe_step_iterations - 1
        size = jnp.where(is_last_step, last_step_size, cfgs.max_steps_ub)

        execute_schedule_chunk(start, size)

    _run()

  num_pre_leaves = 3  # cu_q_lens, kv_lens, page_indices
  num_sched_leaves = len(jax.tree_util.tree_leaves(schedule_hbm))
  ql_nope_hbm_idx = num_pre_leaves + num_sched_leaves
  cache_kv_hbm_idx = ql_nope_hbm_idx + 4

  return pl.pallas_call(
      ragged_paged_attention_pipeline,
      out_shape=[ql_nope_hbm, cache_kv_hbm],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=3,
          in_specs=[
              schedule_hbm.in_specs(),
              pl.BlockSpec(memory_space=pltpu.HBM),  # ql_nope_hbm_ref
              pl.BlockSpec(memory_space=pltpu.HBM),  # q_pe_hbm_ref
              pl.BlockSpec(memory_space=pltpu.HBM),  # new_kv_c_hbm_ref
              pl.BlockSpec(memory_space=pltpu.HBM),  # new_k_pe_hbm_ref
              pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_hbm_ref
          ],
          out_specs=[
              pl.BlockSpec(memory_space=pltpu.HBM),  # aliased_o_hbm_ref
              pl.BlockSpec(memory_space=pltpu.HBM),  # aliased_cache_kv_hbm_ref
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=cfgs.vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      input_output_aliases={ql_nope_hbm_idx: 0, cache_kv_hbm_idx: 1},
      name=get_kernel_name(cfgs),
      metadata=get_kernel_metadata(cfgs),
  )(
      cu_q_lens,
      kv_lens,
      page_indices,
      schedule_hbm,
      ql_nope_hbm,
      q_pe_hbm,
      new_kv_c_hbm,
      new_k_pe_hbm,
      cache_kv_hbm,
  )


# ==============================================================================
# Outermost Kernel Entrypoint
# ==============================================================================


@functools.partial(
    jax.jit,
    static_argnames=(
        "cfgs",
        "debug_mode",
    ),
    donate_argnames=("cache_kv",),
)
def mla_ragged_paged_attention(
    ql_nope: jax.Array,  # [actual_num_q_heads, max_num_tokens, actual_lkv_dim]
    q_pe: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [max_num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [max_num_tokens, actual_r_dim]
    cache_kv: jax.Array,  # [total_num_pages, aligned_kv_dim, page_size]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    cfgs: configs.MlaConfigs,
    schedule_hbm: schedule.MlaSchedule | None = None,
    debug_mode: bool = False,
) -> tuple[
    jax.Array,  # [actual_num_q_heads, max_num_tokens, actual_lkv_dim]
    jax.Array,  # updated_cache_kv: [total_num_pages, aligned_kv_dim, page_size]
]:
  """MLA Ragged paged attention with 3D transposed KV cache support."""
  static_validate_inputs(
      ql_nope,
      q_pe,
      new_kv_c,
      new_k_pe,
      cache_kv,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
      cfgs=cfgs,
  )

  actual_num_q_heads, actual_max_num_tokens, actual_lkv_dim = ql_nope.shape

  if schedule_hbm is None:
    schedule_hbm = schedule.generate_mla_metadata(
        cu_q_lens, kv_lens, page_indices, distribution, cfgs=cfgs
    )

  ql_nope = prepare_q_nope_inputs(
      ql_nope,
      vmem_limit_bytes=cfgs.vmem_limit_bytes,
  )  # [max_num_tokens, num_q_heads, lkv_dim]
  q_pe = prepare_q_inputs(q_pe)  # [max_num_tokens, num_q_heads, r_dim]
  new_kv_c = prepare_kv_inputs_for_transposed_kv_cache(
      new_kv_c,
      page_size=cfgs.serve.page_size,
  )  # [lkv_dim, max_num_tokens]
  new_k_pe = prepare_kv_inputs_for_transposed_kv_cache(
      new_k_pe,
      page_size=cfgs.serve.page_size,
  )  # [r_dim, max_num_tokens]

  if debug_mode:
    logging.info(
        "Prepared inputs for MLA: ql_nope=%s, q_pe=%s, new_kv_c=%s,"
        " new_k_pe=%s, cache_kv=%s",
        ql_nope.shape,
        q_pe.shape,
        new_kv_c.shape,
        new_k_pe.shape,
        cache_kv.shape,
    )

  out_hbm, updated_cache_kv = _mla_ragged_paged_attention_kernel(
      cu_q_lens,
      kv_lens,
      page_indices,
      schedule_hbm,
      ql_nope,
      q_pe,
      new_kv_c,
      new_k_pe,
      cache_kv,
      cfgs=cfgs,
  )

  output = prepare_outputs(
      out_hbm,
      actual_num_q_heads,
      actual_max_num_tokens,
      actual_lkv_dim,
      vmem_limit_bytes=cfgs.vmem_limit_bytes,
  )  # [actual_num_q_heads, max_num_tokens, actual_lkv_dim]

  return output, updated_cache_kv
