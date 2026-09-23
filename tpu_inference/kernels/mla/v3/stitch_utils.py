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
from tpu_inference.kernels.mla.v3 import utils


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
  """Prefill path: roll only the window the new tokens occupy.

  This used to roll the *entire* staging buffer -- `reshape(words, sublanes,
  v_len)` then `pltpu.roll(..., axis=2)` over all `v_len` lanes -- to move at
  most `bkv_sz` tokens. At prefill that is 160 x 8 x 3200 u32 = 16 MB of
  vector traffic per grid step, and a 1024-token prefill at bq_sz=16 is 64
  steps. It is the reason v3's prefill measured 2-3.6x slower than v2's at
  matched shapes.

  The decode path beside this one already targets only the VREG holding the
  stitch boundary. This applies the same idea: source and destination are each
  a 128-aligned window wide enough for the moved run, so the rolled extent is
  `bkv_sz + 128` rather than `v_len`, independent of the slack.

  Returns the merged `[0, bkv_sz)` block, the same extent the caller wrote
  before -- only the rolled *source* shrinks, from `v_len` to `bkv_sz + 128`.
  """
  num_lanes = pltpu.get_tpu_info().num_lanes
  # Destination is exactly the block, [0, bkv_sz) -- the same lanes the old
  # whole-buffer version wrote, so nothing in the slack region (where pending
  # new-KV DMAs land) is touched. Only the *source* needs a window, and it
  # needs `bkv_sz + 128`: the run can be as long as the block, and 128 more
  # covers a source that starts mid-group.
  src_win_len = utils.align_to(cfgs.bkv_sz, num_lanes) + num_lanes
  assert src_win_len <= v_len, (src_win_len, v_len)

  src_tok = cache_pages * cfgs.serve.page_size + new_tok_offset
  src_base = (src_tok // num_lanes) * num_lanes
  src_base = pl.multiple_of(
      jnp.minimum(src_base, v_len - src_win_len), num_lanes
  )

  dst_win = vmem_u32_ref[:, : cfgs.bkv_sz]
  src_win = vmem_u32_ref[:, pl.ds(src_base, src_win_len)]

  # Align the staged run so that lane `bkv_sz_cache` of the rolled window
  # holds the token at `src_tok`. Negative shifts are congruent mod the window
  # width, and `pltpu.roll` requires non-negative.
  shift = (bkv_sz_cache - (src_tok - src_base)) % src_win_len
  rolled = pltpu.roll(src_win, shift, axis=1)[:, : cfgs.bkv_sz]

  lane = jax.lax.broadcasted_iota(jnp.int32, dst_win.shape, 1)
  merged = jax.lax.select(lane >= bkv_sz_cache, rolled, dst_win)
  return merged



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
    # Prefill now returns a window rather than the whole buffer, so the store
    # is a window store. Writing `[..., :bkv_sz]` here used to rewrite every
    # lane of the block on every grid step to land at most `bkv_sz` moved
    # tokens; see `_stitch_prefill_lane`.
    vmem_u32_ref[:, : cfgs.bkv_sz] = stitch_result


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