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

"""Wrapper for MLA v3 kernel to orchestrate multi-phase batch execution."""

import dataclasses

from absl import logging
import jax
from jax import lax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from tpu_inference.kernels.mla.v3 import configs
from tpu_inference.kernels.mla.v3 import kernel
from tpu_inference.kernels.mla.v3 import schedule


# Matches `v2/kernel.py`'s DEFAULT_MASK_VALUE. Deliberately not
# `finfo(float32).min`: the masked logits are fed straight into the online
# softmax, and a value that close to the representable limit turns the
# `s - m` subtraction into -inf (and then 0 * inf -> NaN) for any block whose
# rows are entirely masked. 0.7 of the max leaves room for that subtraction
# while still driving exp() to zero.
DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def calculate_block_sizes(
    serve_cfgs: configs.ServingConfigs,
    *,
    num_kv_pages_per_block: int = 2,
    num_queries_per_block: int = 4,
    batch_size: int = 2,
    n_buffer: int = 2,
    q_split: int = 1,
) -> tuple[configs.BlockSizes, configs.BlockSizes]:
  """Calculates default block sizes for decode and prefill/mixed passes.

  `q_split` divides the prefill/mixed query block into that many sub-chunks by
  setting `bq_c_sz = bq_sz // q_split`. It had been pinned at 1 -- `bq_c_sz`
  was always `num_queries_per_block` -- which made `MlaConfigs.q_split` always
  1, and with it the two-step PV deferral in `chunked_flash_attention` dead
  code everywhere, not just on DECODE. At `q_split == 1` that deferral is
  still a no-op: it flushes the single pending chunk immediately after the
  loop, emitting the same PV in the same order as an inline call.

  Whether splitting the row axis helps depends entirely on how many MXU tiles
  the rows span, and that differs sharply between the two modes. In DECODE
  `n_q = 1 * aligned_num_q_heads = 128` -- exactly one 128x128 MXU tile -- so
  any split divides utilization directly; `head_split` measured +75% at 2 and
  +402% at 8 for precisely that reason. In prefill at `bq_sz = 32`,
  `n_q = 32 * 128 = 4096` rows, or 32 tiles; splitting four ways still leaves
  eight whole tiles per chunk, so the MXU is unaffected while the live score
  tile drops from [1, 4096, S] to a quarter of that.

  Left at 1 for DECODE regardless, where `bq_sz` is already 1.
  """
  page_size = serve_cfgs.page_size
  bkv_sz = num_kv_pages_per_block * page_size

  decode_blocks = configs.BlockSizes(
      bq_sz=1,
      bq_c_sz=1,
      bkv_sz=bkv_sz,
      batch_size=batch_size,
      n_buffer=n_buffer,
  )
  prefill_blocks = configs.BlockSizes(
      bq_sz=num_queries_per_block,
      bq_c_sz=max(1, num_queries_per_block // max(1, q_split)),
      bkv_sz=bkv_sz,
      batch_size=batch_size,
      n_buffer=n_buffer,
  )
  return decode_blocks, prefill_blocks


def mode_configs(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int = 2,
    num_queries_per_block: int = 4,
    batch_size: int = 2,
    n_buffer: int = 2,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    p_same_dtype_as_v: bool = False,
    # None defers to `ServingConfigs.kv_slack_pad_lanes`, which defaults to
    # 128 to keep `kv_vmem_lanes // 128` off a power of two. Hardcoding 0 here
    # silently overrode that default and cost 1.2% on decode.
    kv_slack_pad_lanes: int | None = None,
    q_split: int = 1,
    kv_layout: configs.KVLayout = configs.KVLayout.SEQ_ALONG_LANE,
) -> dict[configs.MlaCase, configs.MlaConfigs]:
  """Returns the `MlaConfigs` each pass runs under, keyed by mode.

  Shared by `mla_ragged_paged_attention` and `build_schedules` so a caller that
  precomputes schedules cannot drift from the configs the kernel then runs
  under -- a mismatch would silently produce a schedule for different block
  sizes.
  """
  actual_num_q_heads, total_q_tokens, actual_lkv_dim = ql_nope.shape
  actual_r_dim = q_pe.shape[-1]
  max_num_seqs = kv_lens.shape[0]
  num_page_indices = page_indices.shape[0]

  if vmem_limit_bytes is None:
    vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
  if mask_value is None:
    mask_value = DEFAULT_MASK_VALUE
  # [pages, kv_dim, page_size] vs [pages, page_size, kv_dim]
  page_size = cache_kv.shape[-1]

  model_cfgs = configs.MlaModelConfigs(
      num_q_heads=actual_num_q_heads,
      lkv_dim=actual_lkv_dim,
      r_dim=actual_r_dim,
      mask_value=mask_value,
      sm_scale=sm_scale,
      soft_cap=soft_cap,
      sliding_window=sliding_window,
  )
  serve_cfgs = configs.ServingConfigs(
      num_seqs=max_num_seqs,
      num_page_indices=num_page_indices,
      total_q_tokens=total_q_tokens,
      dtype_q=ql_nope.dtype,
      dtype_kv=cache_kv.dtype,
      dtype_out=ql_nope.dtype,
      page_size=page_size,
      scale_q=q_scale,
      scale_k=k_scale,
      scale_v=v_scale,
      kv_layout=kv_layout,
      p_same_dtype_as_v=p_same_dtype_as_v,
      **({} if kv_slack_pad_lanes is None
         else dict(kv_slack_pad_lanes=kv_slack_pad_lanes)),
  )
  default_decode, default_prefill = calculate_block_sizes(
      serve_cfgs,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      batch_size=batch_size,
      n_buffer=n_buffer,
      q_split=q_split,
  )

  out = {}
  for mode in (
      configs.MlaCase.DECODE,
      configs.MlaCase.PREFILL,
      configs.MlaCase.MIXED,
  ):
    if mode == configs.MlaCase.DECODE:
      blocks = decode_block_sizes or default_decode
    else:
      blocks = prefill_block_sizes or default_prefill
    # `p_same_dtype_as_v` is a per-pass choice, not a serving-wide one. Keeping
    # P in V's dtype avoids a cast before the PV matmul, which pays off in
    # PREFILL/MIXED where the score tile is large, but on DECODE the tile is a
    # single 128x128 MXU pass and the narrower dtype only costs accuracy.
    pass_serve = dataclasses.replace(
        serve_cfgs,
        p_same_dtype_as_v=(
            False if mode == configs.MlaCase.DECODE else p_same_dtype_as_v
        ),
    )
    out[mode] = configs.MlaConfigs(
        block=blocks,
        model=model_cfgs,
        serve=pass_serve,
        vmem_limit_bytes=vmem_limit_bytes,
        mode=mode,
    )
  return out


def build_schedules(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    distribution: jax.Array,
    cfgs_by_mode: dict[configs.MlaCase, configs.MlaConfigs],
) -> dict[configs.MlaCase, schedule.MlaSchedule]:
  """Builds each pass's schedule once, for reuse across all attention layers."""
  return {
      mode: schedule.generate_mla_metadata(
          cu_q_lens, kv_lens, page_indices, distribution, cfgs=cfgs
      )
      for mode, cfgs in cfgs_by_mode.items()
  }


def mla_ragged_paged_attention(
    ql_nope: jax.Array,  # [actual_num_q_heads, max_num_tokens, actual_lkv_dim] or [max_num_tokens, actual_num_q_heads, actual_lkv_dim]
    q_pe: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [max_num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [max_num_tokens, actual_r_dim]
    cache_kv: jax.Array,  # [total_num_pages, aligned_kv_dim, page_size]
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int = 2,
    num_queries_per_block: int = 4,
    batch_size: int = 2,
    n_buffer: int = 2,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    p_same_dtype_as_v: bool = False,
    # None defers to `ServingConfigs.kv_slack_pad_lanes`, which defaults to
    # 128 to keep `kv_vmem_lanes // 128` off a power of two. Hardcoding 0 here
    # silently overrode that default and cost 1.2% on decode.
    kv_slack_pad_lanes: int | None = None,
    q_split: int = 1,
    kv_layout: configs.KVLayout = configs.KVLayout.SEQ_ALONG_LANE,
    schedules: dict[configs.MlaCase, schedule.MlaSchedule] | None = None,
    debug_mode: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """MLA Ragged paged attention, orchestrated as Decode -> Prefill -> Mixed.

  `schedules` lets the caller supply metadata built by `build_schedules`. The
  schedule depends only on `cu_q_lens`, `kv_lens`, `page_indices`,
  `distribution` and the block sizes -- none of which vary across the model's
  attention layers -- so leaving it to be built here runs the
  `mla_metadata_schedule` kernel once per layer. XLA does not CSE the Pallas
  call, so on a 61-layer model that is 61 launches per step instead of one.
  Left as None it is built inline, which keeps the standalone call sites
  working.
  """
  actual_num_q_heads, total_q_tokens, actual_lkv_dim = ql_nope.shape
  # [pages, kv_dim, page_size] vs [pages, page_size, kv_dim]
  page_size = cache_kv.shape[-1]

  if vmem_limit_bytes is None:
    vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

  cfgs_by_mode = mode_configs(
      ql_nope,
      q_pe,
      cache_kv,
      kv_lens,
      page_indices,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      q_scale=q_scale,
      k_scale=k_scale,
      v_scale=v_scale,
      num_kv_pages_per_block=num_kv_pages_per_block,
      num_queries_per_block=num_queries_per_block,
      batch_size=batch_size,
      n_buffer=n_buffer,
      decode_block_sizes=decode_block_sizes,
      prefill_block_sizes=prefill_block_sizes,
      vmem_limit_bytes=vmem_limit_bytes,
      p_same_dtype_as_v=p_same_dtype_as_v,
      **({} if kv_slack_pad_lanes is None
         else dict(kv_slack_pad_lanes=kv_slack_pad_lanes)),
      q_split=q_split,
      kv_layout=kv_layout,
  )

  init_cfgs = configs.MlaConfigs(
      block=cfgs_by_mode[configs.MlaCase.DECODE].block,
      model=cfgs_by_mode[configs.MlaCase.DECODE].model,
      serve=cfgs_by_mode[configs.MlaCase.DECODE].serve,
      vmem_limit_bytes=vmem_limit_bytes,
      mode=configs.MlaCase.DECODE,
  )
  kernel.static_validate_inputs(
      ql_nope,
      q_pe,
      new_kv_c,
      new_k_pe,
      cache_kv,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
      cfgs=init_cfgs,
  )

  ql_nope_prep = kernel.prepare_q_nope_inputs(
      ql_nope,
      vmem_limit_bytes=vmem_limit_bytes,
  )
  # All three stay at 128-lane alignment. `q_pe` in particular *must*: it is
  # reshaped with `aligned_r_dim` as its minor dimension, which is the lane
  # axis. See `MlaConfigs.kv_dim_align`.
  head_align = init_cfgs.kv_dim_align
  q_pe_prep = kernel.prepare_q_inputs(q_pe, head_align=head_align)
  new_kv_c_prep = kernel.prepare_kv_inputs_for_transposed_kv_cache(
      new_kv_c,
      page_size=page_size,
      head_align=head_align,
  )
  new_k_pe_prep = kernel.prepare_kv_inputs_for_transposed_kv_cache(
      new_k_pe,
      page_size=page_size,
      head_align=head_align,
  )

  def run_mla_kernel(
      mode: configs.MlaCase,
      ql_nope_in: jax.Array,
      cache_kv_in: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    cfgs = cfgs_by_mode[mode]
    if debug_mode:
      logging.info("blocks: %s, mode: %s", cfgs.block, mode)
    if schedules is not None and mode in schedules:
      schedule_hbm = schedules[mode]
    else:
      schedule_hbm = schedule.generate_mla_metadata(
          cu_q_lens, kv_lens, page_indices, distribution, cfgs=cfgs
      )
    return kernel._mla_ragged_paged_attention_kernel(
        cu_q_lens,
        kv_lens,
        page_indices,
        schedule_hbm,
        ql_nope_in,
        q_pe_prep,
        new_kv_c_prep,
        new_k_pe_prep,
        cache_kv_in,
        cfgs=cfgs,
    )

  num_decode = distribution[0]
  num_prefill = distribution[1] - distribution[0]
  num_mixed = distribution[2] - distribution[1]

  if debug_mode:
    logging.info(
        "Prepared inputs for MLA: ql_nope=%s, q_pe=%s, new_kv_c=%s,"
        " new_k_pe=%s, cache_kv=%s",
        ql_nope_prep.shape,
        q_pe_prep.shape,
        new_kv_c_prep.shape,
        new_k_pe_prep.shape,
        cache_kv.shape,
    )

  # `distribution` splits the batch into decode / prefill / mixed bands. Each
  # gets its own pass, chaining `ql_nope_prep` and `cache_kv` so later passes
  # see earlier writes.
  #
  # These used to be guarded by `lax.cond` on whether the band was non-empty,
  # at 0.053 ms per call -- 61 layers deep that is ~3.2 ms per decode step
  # against a ~34 ms TPOT, and it is the single largest v2/v3 difference
  # end to end, larger than every kernel delta combined.
  #
  # The guard was never needed. The kernel is already zero-trip on an empty
  # band: `num_safe_step_iterations = cdiv(actual_steps, max_steps_ub)` is 0
  # and `pl.loop(0, 0)` runs nothing. What the cond cost was its *carry* --
  # both branches must yield `(ql_nope, cache_kv)`, and `cache_kv` is the
  # whole KV cache, so the conditional forced it through a select. v2 reaches
  # the same place by sizing each pass zero-trip and paying ~0.9 us for the
  # passes that do nothing (v2/kernel.py:2511).
  # `distribution` splits the batch into decode / prefill / mixed bands. Each
  # gets its own pass, chaining `ql_nope_prep` and `cache_kv` so later passes
  # see earlier writes.
  #
  # The comment that used to sit here said the cond costs 0.053 ms per call
  # and that v2 avoids it by sizing each pass zero-trip. That was taken at
  # face value and the guard removed; measured as a before/after it is the
  # other way round. Pure decode, kv 9216, bkv=3, batch=4, wall clock:
  #
  #             guard present   guard removed
  #     8 seqs      68.8 us         74.6 us
  #    32 seqs     155.5           163.5
  #   112 seqs     450.3           457.4
  #
  # Running the two empty passes as zero-trip pallas_calls costs 6-8 us more
  # than the conditional it replaced, at every size. v3 already beat v2 with
  # the guard in place (-1.7 / -9.7 / -10.8%). Keep the cond.
  passes = [
      (configs.MlaCase.DECODE, distribution[0] > 0),
      (configs.MlaCase.PREFILL, distribution[1] - distribution[0] > 0),
      (configs.MlaCase.MIXED, distribution[2] - distribution[1] > 0),
  ]

  carry = (ql_nope_prep, cache_kv)
  for mode, predicate in passes:
    carry = lax.cond(
        predicate,
        lambda q_kv, m=mode: run_mla_kernel(m, q_kv[0], q_kv[1]),
        lambda q_kv: q_kv,
        carry,
    )
  o_hbm, cache_kv = carry

  output = kernel.prepare_outputs(
      o_hbm,
      actual_num_q_heads,
      total_q_tokens,
      actual_lkv_dim,
      vmem_limit_bytes=vmem_limit_bytes,
  )

  return output, cache_kv