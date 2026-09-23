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

"""Stitch utilities for Batched MLA v3.

Merges newly-fetched KV tokens into their logical position inside the VMEM
block. The new tokens arrive page-aligned from HBM but belong at
`bkv_sz_cache`, so something has to move them.

Two implementations, because the token axis moves:

  SEQ_ALONG_LANE  -- a token is a *lane*. The block is bitcast to u32 and the
      boundary vreg is rolled and selected lane-wise. O(1) for decode.

      one 32-bit word, so a single token is not addressable in the u32 view at
      all. Instead the stitch works in the native dtype, where one token is a
      contiguous row of `aligned_kv_dim`, and the merge is a row copy.
"""

from typing import Any
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tpu_inference.kernels.mla.v3 import configs


def _stitch_decode_lane(
    vmem_u32_ref: jax.Array,
    bkv_sz_cache: jax.Array,
    cache_pages: jax.Array,
    new_tok_offset: jax.Array,
    v_len: int,
    *,
    cfgs: configs.MlaConfigs,
):
  """O(1) Decode Path: Target exactly the VREG containing the stitch boundary."""
  del v_len
  num_lanes = pltpu.get_tpu_info().num_lanes

  # Destination: VREG chunk index and lane offset for new token insertion.
  dst_chunk_idx = bkv_sz_cache // num_lanes
  dst_rel = bkv_sz_cache % num_lanes

  # Source: VREG chunk index and lane offset of fetched token in VMEM.
  src_tok_idx = cache_pages * cfgs.serve.page_size + new_tok_offset
  src_chunk_idx = src_tok_idx // num_lanes
  src_rel = src_tok_idx % num_lanes

  # Slice directly along lane axis in 3D view: [sublanes, 128]
  dst_vreg = vmem_u32_ref[
      :, pl.ds(pl.multiple_of(dst_chunk_idx * num_lanes, num_lanes), num_lanes)
  ]
  src_vreg = vmem_u32_ref[
      :, pl.ds(pl.multiple_of(src_chunk_idx * num_lanes, num_lanes), num_lanes)
  ]

  rolled_src_vreg = pltpu.roll(src_vreg, dst_rel - src_rel, axis=1)

  lane_idx = jax.lax.broadcasted_iota(jnp.int32, dst_vreg.shape, 1)
  merged_dst_vreg = jax.lax.select(
      lane_idx == dst_rel,
      rolled_src_vreg,
      jnp.where(lane_idx < dst_rel, dst_vreg, 0),
  )

  return dst_chunk_idx, merged_dst_vreg


def _stitch_prefill_lane(
    vmem_u32_ref: jax.Array,
    bkv_sz_cache: jax.Array,
    cache_pages: jax.Array,
    new_tok_offset: jax.Array,
    v_len: int,
    *,
    cfgs: configs.MlaConfigs,
):
  """O(N) Prefill Path: Roll the entire new tokens buffer into place."""
  total_head_words = cfgs.aligned_kv_dim // cfgs.serve.packing_kv
  num_sublanes = pltpu.get_tpu_info().num_sublanes
  words_per_sublane = total_head_words // num_sublanes
  vmem_u32_reshaped = vmem_u32_ref.reshape(
      words_per_sublane, num_sublanes, v_len
  )

  # The `% v_len` is required, not defensive: the difference is negative
  # whenever the new tokens land at a higher lane index than the stitch
  # boundary (`cache_pages` rounds `bkv_sz_cache` *up* to a page, so the source
  # offset routinely exceeds the destination). `pltpu.roll` needs a
  # non-negative shift, and rolling by `d` is congruent to rolling by
  # `d % v_len` on a v_len-wide axis, so the mod maps it to the right one.
  roll_shift = (
      bkv_sz_cache - (cache_pages * cfgs.serve.page_size + new_tok_offset)
  ) % v_len
  rolled_u32 = pltpu.roll(vmem_u32_reshaped[...], roll_shift, axis=2)

  lane_idx = jax.lax.broadcasted_iota(
      jnp.int32, rolled_u32[..., : cfgs.bkv_sz].shape, 2
  )
  merged_cache_u32 = jax.lax.select(
      lane_idx >= bkv_sz_cache,
      rolled_u32[..., : cfgs.bkv_sz],
      vmem_u32_reshaped[..., : cfgs.bkv_sz],
  )

  return merged_cache_u32



# ==============================================================================
# ==============================================================================

def store_new_kv_lane(
    vmem_ref: jax.Ref,
    b_idx: int,
    stitch_result: Any,
    *,
    cfgs: configs.MlaConfigs,
):
  """Stores the result of stitch_new_kv_lane back into memory."""

  vmem_u32_ref = vmem_ref.at[b_idx].bitcast(jnp.uint32)

  if cfgs.one_new_token:
    dst_chunk_idx, merged_dst_vreg = stitch_result
    num_lanes = pltpu.get_tpu_info().num_lanes
    k_chunks = cfgs.serve.page_size // num_lanes
    dst_page_start = (dst_chunk_idx // k_chunks) * k_chunks
    for c_offset in range(k_chunks):
      c = dst_page_start + c_offset
      chunk_slice = pl.ds(
          pl.multiple_of(c * num_lanes, num_lanes), num_lanes
      )
      existing_vreg = vmem_u32_ref[:, chunk_slice]
      new_chunk = jax.lax.select(
          c < dst_chunk_idx, existing_vreg, merged_dst_vreg
      )
      vmem_u32_ref[:, chunk_slice] = new_chunk
  else:
    v_len = cfgs.kv_vmem_lanes
    merged_cache_u32 = stitch_result
    total_head_words = cfgs.aligned_kv_dim // cfgs.serve.packing_kv
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    words_per_sublane = total_head_words // num_sublanes
    vmem_u32_reshaped = vmem_u32_ref.reshape(
        words_per_sublane, num_sublanes, v_len
    )
    vmem_u32_reshaped[..., : cfgs.bkv_sz] = merged_cache_u32


def stitch_new_kv_lane(
    vmem_ref: jax.Ref,
    b_idx: int,
    bkv_sz_frm_cache: jax.Array,
    new_kv_len_start: jax.Array,
    *,
    cfgs: configs.MlaConfigs,
):
  """Fetches and computes stitched KV tokens (separated to avoid RAW hazards).

  Expects vmem_ref shape: [batch, aligned_kv_dim, cfgs.kv_vmem_lanes]
  """
  bkv_sz_cache = bkv_sz_frm_cache.astype(jnp.int32)
  new_tok_offset = new_kv_len_start.astype(jnp.int32) % cfgs.serve.page_size
  cache_pages = pl.cdiv(bkv_sz_cache, cfgs.serve.page_size)

  v_len = cfgs.kv_vmem_lanes
  vmem_u32_ref = vmem_ref.at[b_idx].bitcast(jnp.uint32)

  if cfgs.one_new_token:
    return _stitch_decode_lane(
        vmem_u32_ref,
        bkv_sz_cache,
        cache_pages,
        new_tok_offset,
        v_len,
        cfgs=cfgs,
    )
  else:
    return _stitch_prefill_lane(
        vmem_u32_ref,
        bkv_sz_cache,
        cache_pages,
        new_tok_offset,
        v_len,
        cfgs=cfgs,
    )