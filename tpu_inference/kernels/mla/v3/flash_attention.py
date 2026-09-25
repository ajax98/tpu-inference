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

"""Flash Attention compute primitives for Multi-Head Latent Attention (MLA).

Implements online softmax rescaling, QK dot products (non-positional Q_nope *
C_kv +
RoPE Q_pe * K_pe), causal & sliding window masking, and PV accumulation
(accumulating strictly against latent C_kv without RoPE key dimension).
"""

from collections.abc import Sequence
from typing import Any
import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from tpu_inference.kernels.mla.v3 import configs
from tpu_inference.kernels.mla.v3 import utils


def flash_attention_qk_softmax(
    q: jax.Array,  # [B, bq_sz * H_q, d_q]
    k: jax.Array,  # [B, d_kv, S] (SEQ_ALONG_LANE)
    m_prev: jax.Array,  # [bq_sz * H_q, 128]
    l_prev: jax.Array,  # [bq_sz * H_q, 128]
    is_last_k: jax.Array | Sequence[Any] | None = None,  # [B]
    *,
    processed_q_len: jax.Array | Sequence[jax.Array] | None = None,  # [B]
    processed_kv_len: jax.Array | Sequence[jax.Array] | None = None,  # [B]
    cfgs: configs.MlaConfigs,
    bq_start: int | jax.Array = 0,
) -> tuple[jax.Array, list[jax.Array], jax.Array, jax.Array]:
  """Computes QK matrix multiplications, masking, and online softmax step for MLA.

  MLA QK score is decomposed into:
    S = sm_scale * (Q_nope @ C_kv.T + Q_pe @ K_pe.T)
  which is equivalent to a single dot product of concatenated Q and K.

  Args:
    q: Combined query tensor [Q_nope, Q_pe] of width aligned_q_dim.
    k: Combined key/value tensor [C_kv, K_pe] of width aligned_kv_dim.
    m_prev: Previous running maximum logits for online softmax [bq_sz * H_q,
      128].
    l_prev: Previous running sum of exponentials (denominator) [bq_sz * H_q,
      128].
    is_last_k: Optional boolean flags indicating whether this is the last K
      tile.
    processed_q_len: Sequence start offsets for queries in the batch.
    processed_kv_len: Sequence start offsets for keys in the batch.
    cfgs: MLA configuration parameters.
    bq_start: Block start offset for query within the sequence.

  Returns:
    Tuple of (p, alpha_list, m_carry, l_next).

    `p` is the softmax probability tensor and `alpha_list` the per-batch
    rescaling factors.

    `m_carry` is **not** `m_next`. It is the running max left over after the
    last lane of this block, *after* the end-of-sequence reset: when
    `is_last_k[b]` is set, lane b's carry is forced to `-inf` so the next block
    starts a fresh sequence rather than inheriting the finished one's max. The
    unreset per-lane maxima used to normalize `p` are internal and are not
    returned. `l_next` is the stacked per-lane denominator, and the caller
    keeps only its last lane (`l_next[-1]`) as the carry.
  """
  b = q.shape[0]

  num_q_heads = cfgs.aligned_num_q_heads
  n_q = q.shape[1]  # bq_sz * aligned_num_q_heads
  assert (
      n_q % num_q_heads == 0
  ), f"Q block rows {n_q} not divisible by aligned head count {num_q_heads}"

  # 1. Compute QK dot products: S = Q @ K.T

  s_dim = k.shape[-1]
  s = lax.dot(
      q,
      k,
      dimension_numbers=(([2], [1]), ([0], [0])),
      preferred_element_type=jnp.float32,
  )

  s *= cfgs.model.sm_scale

  if cfgs.serve.scale_k is not None:
    s *= cfgs.serve.scale_k
  if cfgs.serve.scale_q is not None:
    s *= cfgs.serve.scale_q

  # 2. Soft-capping (Gemma / Grok style)
  if cfgs.model.soft_cap is not None:
    s = cfgs.model.soft_cap * jnp.tanh(s / cfgs.model.soft_cap)

  # 3. Causal & Sliding-Window Masking
  if processed_q_len is not None and processed_kv_len is not None:
    sliding_window = cfgs.model.sliding_window

    n_tokens = n_q // num_q_heads

    if n_tokens == 1:
      # One token per block, so q_iota is identically 0 and the predicate
      # collapses to `kv_iota <= -offset`: a [1, s_dim] iota broadcast over
      # rows. The general path below builds a full [n_q, s_dim] iota and an
      # int32 divide per step to produce a tile of zeros. -3.4% on decode.
      kv_iota_1d = lax.broadcasted_iota(jnp.int32, (1, s_dim), 1)
      s_masked = []
      for b_idx in range(b):
        offset = processed_kv_len[b_idx] - (bq_start + processed_q_len[b_idx])
        mask_b = kv_iota_1d <= -offset
        if sliding_window is not None:
          # -kv_iota < sliding_window + offset <=> kv_iota > -offset - sw
          mask_b = jnp.logical_and(
              mask_b, kv_iota_1d > -offset - sliding_window
          )
        s_masked.append(jnp.where(mask_b, s[b_idx], cfgs.model.mask_value))
      s = jnp.stack(s_masked, axis=0)
    else:
      q_iota = lax.broadcasted_iota(jnp.int32, (n_tokens, 1), 0)
      kv_iota = lax.broadcasted_iota(jnp.int32, (1, s_dim), 1)
      q_kv_diff = q_iota - kv_iota

      s_masked = []
      for b_idx in range(b):
        offset = processed_kv_len[b_idx] - (bq_start + processed_q_len[b_idx])
        mask_b = q_kv_diff >= offset

        if sliding_window is not None:
          mask_b = jnp.logical_and(mask_b, q_kv_diff < sliding_window + offset)

        mask_b = jnp.expand_dims(mask_b, axis=1)
        s_b = s[b_idx].reshape(n_tokens, num_q_heads, s_dim)
        s_masked.append(
            jnp.where(mask_b, s_b, cfgs.model.mask_value).reshape(n_q, s_dim)
        )
      s = jnp.stack(s_masked, axis=0)

  # 4. Online Softmax Running Statistics
  s_curr_max = jnp.max(s, axis=-1, keepdims=True)

  alpha_list = []
  m_next_list = []

  for b_idx in range(b):
    m_curr_b = s_curr_max[b_idx]
    m_next_b = jnp.maximum(m_prev, m_curr_b)
    alpha_b = jnp.exp(m_prev - m_next_b)
    alpha_list.append(alpha_b)
    m_next_list.append(m_next_b)
    if is_last_k is not None:
      m_prev = jnp.where(is_last_k[b_idx], -jnp.inf, m_next_b)
    else:
      m_prev = m_next_b

  m_next = jnp.stack(m_next_list, axis=0)

  # 5. Softmax Probabilities
  p = jnp.exp(s - utils.broadcast_minor(m_next, s.shape))
  p_rowsum = jnp.sum(p, axis=-1, keepdims=True, dtype=jnp.float32)

  l_next_list = []
  for b_idx in range(b):
    l_prev_b = l_prev
    l_next_b = alpha_list[b_idx] * l_prev_b + p_rowsum[b_idx]

    l_next_list.append(l_next_b)
    l_prev = l_next_b

  l_next = jnp.stack(l_next_list, axis=0)

  # `m_prev` here is the post-reset carry for the next block, not `m_next`.
  m_carry = m_prev
  return p, alpha_list, m_carry, l_next


def flash_attention_pv(
    p: jax.Array,  # [B, bq_sz * H_q, S]
    v: jax.Array,  # [B, d_nope, S]
    alpha_list: Sequence[jax.Array],  # B * [bq_sz * H_q, 1]
    o_prev: jax.Array,  # [bq_sz * H_q, d_nope]
    cfgs: configs.MlaConfigs,
) -> jax.Array:
  """Accumulates P @ V where V is strictly latent C_kv (excluding RoPE key K_pe).

  Args:
    p: Softmax probabilities tensor [B, bq_sz * H_q, S].
    v: Latent KV cache tensor (C_kv) with width d_nope (e.g. 512).
    alpha_list: Per-batch rescaling factor to update previous accumulator.
    o_prev: Previous unnormalized accumulator [bq_sz * H_q, d_nope].
    cfgs: MLA configuration parameters.

  Returns:
    Updated unnormalized output accumulator [B, bq_sz * H_q, d_nope].
  """
  b = p.shape[0]

  # Narrows the PV operand, not the accumulator -- `preferred_element_type`
  # below still accumulates in f32. -16.1% on chunked_prefill_f8_kv8192,
  # +9.4% on decode, so it stays per-shape.
  if cfgs.serve.p_same_dtype_as_v:
    p = p.astype(v.dtype)

  # p: [b, n_q, s]. Contract s against v's token axis, which is dim 2 under
  # SEQ_ALONG_LANE ([b, d_nope, s])
  # ([b, s, d_nope]). Result is [b, n_q, d_nope] either way.
  pv = lax.dot(
      p,
      v,
      dimension_numbers=(([2], [2]),
                         ([0], [0])),
      preferred_element_type=jnp.float32,
  )
  if cfgs.serve.scale_v is not None:
    pv *= cfgs.serve.scale_v

  o_next_list = []
  for b_idx in range(b):
    alpha_b = utils.broadcast_minor(alpha_list[b_idx], o_prev.shape)
    o_next_b = alpha_b * o_prev + pv[b_idx]
    o_next_list.append(o_next_b)
    o_prev = o_next_b

  return jnp.stack(o_next_list, axis=0)


def chunked_flash_attention(
    q: Any,  # Ref or Array [B, bq_sz * H_q, d_q]
    k: Any,  # Ref or Array [B, d_kv, S]
    m_scratch_ref: Any,  # Ref or Array [bq_sz * H_q, 128]
    l_scratch_ref: Any,  # Ref or Array [bq_sz * H_q, 128]
    acc_scratch_ref: Any,  # Ref or Array [bq_sz * H_q, d_nope]
    o_vref: Any,  # Ref or Array [bq_sz * H_q, d_nope]
    is_last_k: jax.Array | Sequence[Any] | None = None,  # [B]
    *,
    processed_q_len: jax.Array | Sequence[jax.Array] | None = None,
    processed_kv_len: jax.Array | Sequence[jax.Array] | None = None,
    cfgs: configs.MlaConfigs,
) -> None:
  """Executes Flash Attention chunked into bq_c_sz sub-blocks along the query dimension.

  Splits the query ref of size bq_sz into q_split sub-chunks of size bq_c_sz
  (q_split = bq_sz // bq_c_sz) to reduce VMEM register pressure during GEMM
  execution.

  Args:
    q: Full combined query ref or array [Q_nope, Q_pe] [B, bq_sz * H_q, aligned_q_dim].
    k: Full combined key ref or array [C_kv, K_pe] [B, aligned_kv_dim, S].
    m_scratch_ref: Scratch ref for running max logits [bq_sz * H_q, 128].
    l_scratch_ref: Scratch ref for running sum of exponentials [bq_sz * H_q,
      128].
    acc_scratch_ref: Scratch ref for output accumulator [bq_sz * H_q, d_nope].
    o_vref: Output accumulator ref [bq_sz * H_q, d_nope].
    is_last_k: Optional boolean flags indicating whether this is the last K
      tile.
    processed_q_len: Sequence start offsets for queries in the batch.
    processed_kv_len: Sequence start offsets for keys in the batch.
    cfgs: MLA configuration parameters (containing bq_sz and bq_c_sz).

  Returns:
    Tuple of (m_carry, l_next, o_next).

    Note the asymmetry, inherited from `flash_attention_qk_softmax`: `m_carry`
    is rank-2 -- the single post-reset running max left after the *last* batch
    lane -- whereas `l_next` and `o_next` are stacked per-lane and carry a
    leading [B] axis. Callers therefore store `m_carry` directly but must take
    `l_next[-1]` / `o_next[-1]` to get the corresponding carries.
  """
  q_split = cfgs.q_split
  total_q = q.shape[1]
  q_chunk_len = total_q // q_split
  bq_sz_chunk = cfgs.bq_c_sz

  k_arr = k[...]
  k_nope_arr = k_arr[:, :cfgs.aligned_lkv_dim, :]

  def calculate_and_store_out(
      is_last_k: jax.Array | Sequence[Any] | None,
      acc: jax.Array,
      l_val: jax.Array,
      o_vref: jax.Ref,
      *,
      cfgs: configs.MlaConfigs,
      prev_bq_start: int,
  ) -> None:
    """Normalizes accumulated attention output by denominator l and stores to o_vref."""

    def _accum(
        b_idx: int | jax.Array, batch_acc: jax.Array, batch_l: jax.Array
    ):
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
      prev_bq_end = prev_bq_start + bq_sz_chunk
      out = out.reshape(
          cfgs.block.bq_c_sz, cfgs.aligned_num_q_heads, cfgs.aligned_lkv_dim
      )
      o_vref[b_idx, prev_bq_start:prev_bq_end, ...] = out

    for b in range(cfgs.batch_size):
      _accum(b, acc[b], l_val[b])
    # if cfgs.fuse_accum:

    # else:
    #   for b in range(cfgs.batch_size):
    #     assert is_last_k is not None
    #     is_last_k_b = is_last_k[b] == 1
    #     acc_val = acc[b]
    #     l_v = l_val[b]
    #     accum_named_call = jax.named_call(_accum, name=f"accum_{b}")
    #     jax.lax.cond(
    #         is_last_k_b,
    #         accum_named_call,
    #         lambda *_: None,
    #         b,
    #         acc_val,
    #         l_v,
    #     )

  prev_p, prev_alpha, prev_l_next = None, None, None
  prev_q_start, prev_q_end, prev_bq_start = None, None, None

  for q_idx in range(q_split):
    start = q_idx * q_chunk_len
    end = start + q_chunk_len
    bq_start = q_idx * bq_sz_chunk

    q_chunk = q[:, start:end][...]
    m_scratch_chunk = m_scratch_ref[start:end][...]
    l_scratch_chunk = l_scratch_ref[start:end][...]

    # Step 1: Compute QK + Softmax for current chunk
    cur_p, cur_alpha, cur_m_carry, cur_l_next = flash_attention_qk_softmax(
        q_chunk,
        k_arr,
        m_scratch_chunk,
        l_scratch_chunk,
        is_last_k=is_last_k,
        processed_q_len=processed_q_len,
        processed_kv_len=processed_kv_len,
        cfgs=cfgs,
        bq_start=bq_start,
    )
    m_scratch_ref[start:end] = cur_m_carry
    l_scratch_ref[start:end] = cur_l_next[-1]

    # Step 2: Compute PV for previous chunk in parallel with VALU ops
    if prev_p is not None:
      assert prev_alpha is not None
      assert prev_l_next is not None
      assert prev_q_start is not None
      assert prev_q_end is not None
      assert prev_bq_start is not None

      batched_output = flash_attention_pv(
          prev_p,
          k_nope_arr,
          prev_alpha,
          acc_scratch_ref[prev_q_start:prev_q_end][...],
          cfgs=cfgs,
      )
      if cfgs.batch_size > 1:
        calculate_and_store_out(
            is_last_k,
            batched_output,
            prev_l_next,
            o_vref,
            cfgs=cfgs,
            prev_bq_start=prev_bq_start,
        )
      acc_scratch_ref[prev_q_start:prev_q_end] = batched_output[-1]

    prev_p, prev_alpha, prev_l_next = cur_p, cur_alpha, cur_l_next
    prev_q_start, prev_q_end, prev_bq_start = start, end, bq_start

  # Flush the final chunk's PV
  if prev_p is not None:
    assert prev_alpha is not None
    assert prev_l_next is not None
    assert prev_q_start is not None
    assert prev_q_end is not None
    assert prev_bq_start is not None

    batched_output = flash_attention_pv(
        prev_p,
        k_nope_arr,
        prev_alpha,
        acc_scratch_ref[prev_q_start:prev_q_end][...],
        cfgs=cfgs,
    )
    if cfgs.batch_size > 1:
      calculate_and_store_out(
          is_last_k,
          batched_output,
          prev_l_next,
          o_vref,
          cfgs=cfgs,
          prev_bq_start=prev_bq_start,
      )
    acc_scratch_ref[prev_q_start:prev_q_end] = batched_output[-1]

  # For batch_size == 1 (e.g. prefill), perform normalization and store
  # once at the end of the step, avoiding per-chunk stores and branches.
  if cfgs.batch_size == 1:
    def _write_out_b0():
      exact_div = (
          cfgs.serve.dtype_out == jnp.float32
          or cfgs.serve.dtype_out == l_scratch_ref.dtype == jnp.bfloat16
      )
      for i in range(cfgs.q_split):
        start = i * (cfgs.bq_c_sz * cfgs.aligned_num_q_heads)
        end = start + (cfgs.bq_c_sz * cfgs.aligned_num_q_heads)
        bq_start = i * cfgs.bq_c_sz
        bq_end = bq_start + cfgs.bq_c_sz

        chunk_acc = acc_scratch_ref[start:end][...]
        chunk_l = utils.broadcast_minor(
            l_scratch_ref[start:end][...], chunk_acc.shape
        )
        if exact_div:
          chunk_res = lax.div(chunk_acc, chunk_l)
        else:
          chunk_res = chunk_acc * pl.reciprocal(chunk_l, approx=True)
        chunk_out = chunk_res.astype(cfgs.serve.dtype_out).reshape(
            cfgs.bq_c_sz, cfgs.aligned_num_q_heads, cfgs.aligned_lkv_dim
        )
        o_vref[0, bq_start:bq_end, ...] = chunk_out

    if is_last_k is not None:
      is_last = (
          is_last_k[0] if isinstance(is_last_k, (list, tuple)) else is_last_k
      )
      if isinstance(is_last, bool):
        if is_last:
          _write_out_b0()
      else:
        jax.lax.cond(is_last == 1, _write_out_b0, lambda: None)
    else:
      _write_out_b0()
