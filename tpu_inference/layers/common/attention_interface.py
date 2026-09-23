# Copyright 2025 Google LLC
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

import functools
import math
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention
from jax.experimental.pallas.ops.tpu.splash_attention import \
    splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import \
    splash_attention_mask as mask_lib
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import Sharding

import tpu_inference.kernels.ragged_paged_attention.v3.kernel_hd64 as rpa_hd64
from tpu_inference import envs
from tpu_inference.kernels.flash_attention.kernel import flash_attention
import tpu_inference.kernels.mla.v3.configs as mla_v3_configs
import tpu_inference.kernels.mla.v3.mla_wrapper as mla_v3_wrapper
from tpu_inference.kernels.mla import version as mla_version
from tpu_inference.kernels.mla.v2.kernel import mla_ragged_paged_attention
from tpu_inference.kernels.mla.v3.mla_wrapper import (
    mla_ragged_paged_attention as mla_ragged_paged_attention_v3)
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_megacore, get_mesh_shape_product

logger = init_logger(__name__)

MAX_ALLOWED_PAGE_INDICES_N = (
    128 * 1024
)  # Based on experiments on v5e, 256x1024 results in smem oom but 128x1024 not. TODO: Adjust this based on TPU version.

# NOTE: this kernel is experimental and not fully tested.  See
# tpu-inference/tpu_inference/kernels/experimental/batched_rpa/wrapper.py
# for details
if envs.USE_BATCHED_RPA_KERNEL:
    import tpu_inference.kernels.experimental.batched_rpa.wrapper as rpa
    logger.info_once("Using experimental batched RPA kernel")
else:
    import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa
    logger.info_once("Using default RPA kernel")

ragged_paged_attention = rpa.ragged_paged_attention
get_kv_cache_shape = rpa.get_kv_cache_shape

ragged_paged_attention_hd64 = rpa_hd64.ragged_paged_attention_hd64
get_kv_cache_shape_hd64 = rpa_hd64.get_kv_cache_shape


def sharded_flash_attention(
    mesh: Mesh,
    causal: bool = True,
    sm_scale: Optional[float] = None,
    vmem_limit_bytes: int | None = None,
    use_attention_bias: bool = False,
) -> Callable[..., Any]:
    if use_attention_bias:
        in_specs = (
            P("data", "model", None, None),  # q
            P("data", "model", None, None),  # k
            P("data", "model", None, None),  # v
            P("data", "model", None, None),  # attention_bias
            P("data", None),  # segment_ids (B matches q's B, so shard 'data')
        )
        out_specs = P("data", "model", None, None)

        def _flash_attention_use_ab(q, k, v, attention_bias, segment_ids):
            return flash_attention(q,
                                   k,
                                   v,
                                   ab=attention_bias,
                                   segment_ids=segment_ids,
                                   sm_scale=sm_scale,
                                   causal=causal,
                                   vmem_limit_bytes=vmem_limit_bytes)

        attn_fn = _flash_attention_use_ab
    else:
        in_specs = (
            P("data", "model", None, None),  # q
            P("data", "model", None, None),  # k
            P("data", "model", None, None),  # v
            P("data", None),  # segment_ids (B matches q's B, so shard 'data')
        )
        out_specs = P("data", "model", None, None)

        def _flash_attention(q, k, v, segment_ids):
            return flash_attention(q,
                                   k,
                                   v,
                                   segment_ids=segment_ids,
                                   sm_scale=sm_scale,
                                   causal=causal,
                                   vmem_limit_bytes=vmem_limit_bytes)

        attn_fn = _flash_attention

    return jax.jit(
        jax.shard_map(attn_fn,
                      mesh=mesh,
                      in_specs=in_specs,
                      out_specs=out_specs,
                      check_vma=False))


def sharded_paged_attention(
    mesh: Mesh,
    attn_logits_soft_cap: Optional[float] = None,
) -> Callable[..., Any]:
    """Shards GQA PagedAttention along KV heads."""
    in_specs = (
        P(None, "model", None),  # q
        P("model", None, None, None),  # k
        P("model", None, None, None),  # v
        P(),  # lengths
        P(),  # page_indices
    )
    out_specs = P(None, "model", None)

    def _paged_attention_fn(q, k, v, lengths, page_indices):
        if page_indices.size > MAX_ALLOWED_PAGE_INDICES_N:
            raise ValueError(
                "This will result in smem OOM. Use `paged_attention_with_guarded_smem` to run with minibatches."
            )
        return paged_attention(
            q,
            k,
            v,
            lengths,
            page_indices,
            attn_logits_soft_cap=attn_logits_soft_cap,
            pages_per_compute_block=min(
                16, page_indices.shape[1]),  # 512 / page_size:32,
            megacore_mode="kv_head" if get_megacore() else None,
        )

    return jax.jit(
        jax.shard_map(
            _paged_attention_fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        ))


# TODO(xiangxu): merge this with sharded_paged_attention
@jax.jit(static_argnames=["paged_attention_kernel"])
def paged_attention_with_guarded_smem(
    paged_attention_kernel: Callable,
    q: jax.Array,
    k_pages: jax.Array,
    v_pages: jax.Array,
    lengths: jax.Array,
    page_indices: jax.Array,
):
    # Addresses b/336316706. Summary:
    # Paged attention kernel stores `lengths` (batch_size * 4 bytes) and `page_indices` (batch_size * num_blocks_per_seq * 4 bytes) in SMEM.
    # Capacity of SMEM is quite limited which is also TPU version dependent. Models with higher context length or higher batch size, can cause OOM in SMEM.
    # There are two solutions:
    # 1. Reduce blocks per seq by increasing page size.
    # 2. Splitting the batch into several minibatches (Higher perf based on my benchmark).

    batch_size, blocks_per_seq = page_indices.shape

    if page_indices.size <= MAX_ALLOWED_PAGE_INDICES_N:
        return paged_attention_kernel(q, k_pages, v_pages, lengths,
                                      page_indices)

    mini_batch_size = MAX_ALLOWED_PAGE_INDICES_N // blocks_per_seq

    # If batch_size is not disible by mini_batch_size,
    # we set mini_batch_size to a smaller value, i.e GCD,
    # which will trigger more kernel launches but it's fine.
    # TODO: Fix --decode_seqs_padding with this limitation.
    mini_batch_size = math.gcd(batch_size, mini_batch_size)

    num_kernel_launches = batch_size // mini_batch_size

    outputs = jnp.zeros_like(q).reshape(
        (num_kernel_launches, mini_batch_size, *q.shape[1:]))
    q = q.reshape((num_kernel_launches, mini_batch_size, *q.shape[1:]))
    seq_lens = lengths.reshape((num_kernel_launches, mini_batch_size))
    block_indices = page_indices.reshape(
        (num_kernel_launches, mini_batch_size, page_indices.shape[1]))

    for i in range(num_kernel_launches):
        outputs = outputs.at[i].set(
            paged_attention_kernel(q[i], k_pages, v_pages, seq_lens[i],
                                   block_indices[i]))

    outputs = outputs.reshape((batch_size, *outputs.shape[2:]))

    return outputs


# ruff: noqa: E741
def update_cache(
    is_prefill,
    cache,
    indices,
    operand,
    prefill_seq_len=None,
    sliding_window=None,
) -> jax.Array:

    # (8, 55640, 32, 128) (1, 8, 256, 128) -> K (8, 8, 32, 128)
    # I = B * T // S
    # k cache, operand

    B, K, T, H = operand.shape
    K_c, L, S, H = cache.shape
    assert K == K_c
    # NOTE: The cache updating is pretty tricky:
    # 1. The random access updating cache is not as performant as the slice updating.
    #    If the random access is necessary, make sure the indexing count is as small as possible.
    # 2. The random access updating may trigger extra tranpose (memory copy) of cache,
    #    which is a disaster because the cache is huge. This is a data formatting op inserted by
    #    the XLA compiler and not well documented.
    # To mitigate the issues above:
    # For prefill:
    # We reshape the operand so that we can update the cache in block wise, which only requires the block indices.
    # For decode:
    # We reshape the cache so that we can update the cache in token wise, which only requires the token indices (block_id + offset).
    if is_prefill:
        # In the case of sliding window, we should select sliding_window tokens from actual prompt, not from the padded tokens.
        if sliding_window and T > sliding_window:
            assert B == 1
            start_index = jax.lax.max(0, prefill_seq_len - sliding_window)
            operand = jax.lax.dynamic_slice_in_dim(
                operand, start_index, sliding_window,
                axis=2)  # TODO: @pooyam Perf check this.
            T = sliding_window

        I = B * T // S
        # cache: (K, L, S, H)
        # operand: (B, K, T, H) -> (K, I, S, H)
        # indices: (B, T // S) -> (I,)
        operand = jnp.swapaxes(operand, 0, 1).reshape(K, I, S, H)
        indices = indices.reshape(I)
        cache = cache.at[:, indices, :, :].set(operand)
    else:
        # cache: (K, L, S, H) -> (K, L * S, H)
        # operand: (B, K, 1, H) -> (K, B, H)
        # indices: (B,)
        cache = cache.reshape(K, L * S, H)
        operand = jnp.swapaxes(operand, 0, 1).reshape(K, B, H)
        # NOTE: `cache.[:, indices, :].set()` will trigger the extra tranpose of the cache.
        # The `jnp.arange(K)[..., None]` trick is to avoid it. WTF?
        cache = cache.at[jnp.arange(K)[..., None], indices, :].set(operand)
        cache = cache.reshape(K, L, S, H)
    return cache


@jax.jit(static_argnames=["window_size", "attn_logits_soft_cap", "is_mqa"])
def apply_splash(q, k, v, window_size, attn_logits_soft_cap,
                 is_mqa) -> jax.Array:
    # q: (batch_size, num_heads, seq_len, head_dim)
    num_heads = q.shape[1]
    q_seq_len = q.shape[2]
    kv_seq_len = k.shape[2]
    assert kv_seq_len >= q_seq_len

    masks = [
        mask_lib.LocalMask((q_seq_len, kv_seq_len), (window_size, 0),
                           kv_seq_len - q_seq_len) for _ in range(num_heads)
    ]
    mask = mask_lib.MultiHeadMask(tuple((m for m in masks)))
    block_sizes = splash.BlockSizes.get_default()

    if is_mqa:
        attn = splash.make_splash_mqa_single_device(
            mask,
            block_sizes=block_sizes,
            attn_logits_soft_cap=attn_logits_soft_cap)
    else:
        attn = splash.make_splash_mha_single_device(
            mask,
            block_sizes=block_sizes,
            attn_logits_soft_cap=attn_logits_soft_cap)
    attn = jax.vmap(attn)
    outputs = attn(q, k, v, None)

    return outputs


def sharded_splash_attention(
    mesh: Mesh,
    window_size: Optional[int] = None,
    attn_logits_soft_cap: Optional[float] = None,
    is_mqa: bool = False,
) -> Callable[..., Any]:
    in_specs = (
        P("data", "model", None, None),  # q
        P("data", "model", None, None),  # k
        P("data", "model", None, None),  # vx
    )
    out_specs = P("data", "model", None, None)
    return jax.jit(
        jax.shard_map(
            functools.partial(
                apply_splash,
                window_size=window_size,
                attn_logits_soft_cap=attn_logits_soft_cap,
                is_mqa=is_mqa,
            ),
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        ))


def sharded_ragged_paged_attention(
    mesh: Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sink: jax.Array | None,
    sm_scale: float,
    attention_chunk_size: int | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    update_kv_cache: bool = True,
):
    """Shards along KV heads."""
    # Handle GQA/MQA where num_kv_heads < tp_size
    # We replicate KV heads to match tp_size so that we can shard them evenly.
    # TODO (ranlihao): This is not performant and introduces extra overhead during inference. We need to handle this during weight loading
    tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)
    if tp_size > 1:
        num_kv_heads = k.shape[1]
        if num_kv_heads < tp_size:
            if tp_size % num_kv_heads != 0:
                raise ValueError(
                    f"For GQA/MQA, tp_size {tp_size} must be divisible by num_kv_heads {num_kv_heads}"
                )
            factor = tp_size // num_kv_heads
            k = jnp.repeat(k, factor, axis=1)
            v = jnp.repeat(v, factor, axis=1)

    qkv_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None)
    kv_cache_spec = P(ShardingAxisName.ATTN_DATA, None,
                      ShardingAxisName.ATTN_HEAD, None, None)
    in_specs = (
        qkv_spec,  # q
        qkv_spec,  # k
        qkv_spec,  # v
        kv_cache_spec,  # kv cache
        P(ShardingAxisName.ATTN_DATA),  # kv_lens
        P(ShardingAxisName.ATTN_DATA),  # page_indices
        P(ShardingAxisName.ATTN_DATA),  # cu_q_lens
        P(ShardingAxisName.ATTN_DATA),  # distribution
    )
    out_specs = (qkv_spec, kv_cache_spec)

    args = (q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution)

    use_hd64 = q.shape[-1] == 64
    func = ragged_paged_attention_hd64 if use_hd64 else ragged_paged_attention

    if attention_sink is not None:
        if not use_hd64:
            raise NotImplementedError(
                "Attention sink support is only available when head_dim==64")

        in_specs += (P(ShardingAxisName.ATTN_HEAD), )
        args += (attention_sink, )

    # update_kv_cache=False (KV-share) is supported by the v3 default RPA
    # kernel and by the experimental batched RPA kernel. The hd64 path
    # doesn't accept it; fail loud rather than silently ignoring.
    if use_hd64 and not update_kv_cache:
        raise NotImplementedError(
            "update_kv_cache=False (KV-share) is not supported on the "
            "head_dim==64 RPA kernel.")

    def _ragged_paged_attention(*args):
        kwargs = dict(
            sm_scale=sm_scale,
            sliding_window=attention_chunk_size,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        # update_kv_cache is supported by both the v3 default and batched
        # RPA kernels; only the hd64 path doesn't accept it. Default True
        # is a no-op so we don't forward it to the hd64 signature.
        if not use_hd64:
            kwargs["update_kv_cache"] = update_kv_cache
        return func(*args, **kwargs)

    return jax.shard_map(
        _ragged_paged_attention,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(*args)


def attention(
    kv_cache: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_metadata: AttentionMetadata,
    mesh: Mesh,
    head_dim_original: int | None = None,  # before padding,
    sm_scale: float | None = None,
    attention_chunk_size: int | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    sinks: jax.Array | None = None,
    update_kv_cache: bool = True,
) -> Tuple[jax.Array, jax.Array]:
    # T: seq_len
    # N: num_heads
    # K: num_kv_heads
    # D: hidden_size
    # H: head_dim
    # L: num_blocks
    # S: block_size

    # TODO(jevinjiang, cuiq): transpose q weight offline.
    # q: (T, N, H)
    # k,v: (T, K, H)

    if head_dim_original is None:
        head_dim_original = q.shape[-1]

    if sm_scale is None:
        sm_scale = head_dim_original**-0.5

    md = attention_metadata

    # (T, N, H)
    output, kv_cache = sharded_ragged_paged_attention(
        mesh,
        q,
        k,
        v,
        kv_cache,
        md.seq_lens,
        md.block_tables,
        md.query_start_loc,
        md.request_distribution,
        sinks,
        sm_scale=sm_scale,
        attention_chunk_size=attention_chunk_size,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        update_kv_cache=update_kv_cache,
    )

    return kv_cache, output


# v2's production block sizes, as (decode, prefill, mixed). v3 is held to the
# same tiling so a v2-vs-v3 delta reflects the kernel, not the tuning.
_MLA_KV_PAGES_PER_BLOCK = (3, 1, 1)
_MLA_QUERIES_PER_BLOCK = (1, 16, 16)
_MLA_DECODE_BATCH_SIZE = 4
# v3 decode matches v2's tiling, which measured as v3's optimum too.
#
# At the shape E2E actually decodes at -- 9 pages/seq (kv 9216), 16 q tokens,
# from the HLO's page_indices s32[1008] over 112 slots -- sweeping bkv x batch
# with 16 kernel calls per dispatch (so device time dominates), interleaved,
# 5 reps:
#
#     bkv=3 batch=4   80.48 us  sd 0.49   <- this
#     bkv=3 batch=2   85.79      0.26
#     bkv=9 batch=2   86.87      2.23
#     bkv=2 batch=8   90.32      0.62     <- previous setting, +10.9%
#     bkv=2 batch=1  142.15     11.54
#
# The driver is partial-tail waste, not grid-step count: bkv must divide the
# sequence length in pages or the last block carries real tokens in a mostly
# empty slot. 9/3 is exact; 9/2 leaves a half-empty fifth block. The same
# sweep at kv 4096 (4 pages) picked bkv=4 for the same reason, and that is
# also why bkv=4 lost in E2E when tried earlier -- 4 does not divide 9.
_MLA_V3_DECODE_KV_PAGES = _MLA_KV_PAGES_PER_BLOCK[0]
_MLA_V3_DECODE_N_BUFFER = 2
_MLA_V3_DECODE_BATCH = _MLA_DECODE_BATCH_SIZE

# Keyed on the identity of the metadata arrays, which are rebuilt every step.
# Only ever holds the current step.
_V3_SCHEDULE_CACHE: dict = {}


def _v3_kernel_kwargs(page_size: int) -> dict:
    """Block sizes for v3, mirroring v2's per-mode tuples.

    v2 takes (decode, prefill, mixed) tuples; v3 takes a BlockSizes per pass, so
    these are built explicitly rather than via v3's scalar defaults, which would
    force a single num_kv_pages_per_block on both modes.
    """
    return dict(
        decode_block_sizes=mla_v3_configs.BlockSizes(
            bq_sz=1,
            bq_c_sz=1,
            # Left at v2's values deliberately. Sweeping bkv (1-6), batch
            # (1-8) and n_buffer (2-4) at 8 and 112 seqs/device produced
            # apparent wins of +2.0%, +6.1% and +5.0%, none of which survived a
            # controlled A/B: interleaved repeats put n_buffer=4 at +0.23%
            # against a 3.18% within-config sd, and the identical shipped config
            # varied 0.3392 -> 0.3261 ms between jobs.
            #
            # Single-shot medians compared across jobs cannot resolve anything
            # below ~4% here. There is no decode tiling win above that floor.
            bkv_sz=_MLA_V3_DECODE_KV_PAGES * page_size,
            batch_size=_MLA_V3_DECODE_BATCH,
            n_buffer=_MLA_V3_DECODE_N_BUFFER,
        ),
        prefill_block_sizes=mla_v3_configs.BlockSizes(
            bq_sz=_MLA_QUERIES_PER_BLOCK[1],
            bq_c_sz=_MLA_QUERIES_PER_BLOCK[1],
            bkv_sz=_MLA_KV_PAGES_PER_BLOCK[1] * page_size,
            batch_size=1,
            n_buffer=2,
        ),
        # Keeping P in V's dtype skips a cast before the PV matmul. The wrapper
        # applies it to PREFILL/MIXED only -- on DECODE the score tile is a
        # single MXU pass, so it buys nothing.
        p_same_dtype_as_v=True,
    )


def _v3_schedules_for_step(md, mesh, in_specs, q_NTA, q_rope_TNH, kv_cache,
                           sm_scale, q_scale, k_scale, v_scale):
    """Builds v3's per-pass schedules once per step, shared by every layer."""
    key = (id(md.seq_lens), id(md.block_tables), id(md.query_start_loc))
    hit = _V3_SCHEDULE_CACHE.get(key)
    if hit is not None:
        return hit

    def _gen(q, q_rope, cache, kv_lens, page_indices, cu_q_lens, distribution):
        cfgs = mla_v3_wrapper.mode_configs(
            q,
            q_rope,
            cache,
            kv_lens,
            page_indices,
            sm_scale=sm_scale,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            **_v3_kernel_kwargs(cache.shape[-1]),
        )
        return mla_v3_wrapper.build_schedules(cu_q_lens, kv_lens, page_indices,
                                              distribution, cfgs)

    # Same specs the kernel call uses for these operands, so the schedule is
    # built over the identical shards.
    gen_in_specs = (in_specs[0], in_specs[1], in_specs[4], in_specs[5],
                    in_specs[6], in_specs[7], in_specs[8])
    out = jax.jit(
        jax.shard_map(_gen,
                      mesh=mesh,
                      in_specs=gen_in_specs,
                      out_specs=P(ShardingAxisName.ATTN_DATA),
                      check_vma=False))(q_NTA, q_rope_TNH, kv_cache,
                                        md.seq_lens, md.block_tables,
                                        md.query_start_loc,
                                        md.request_distribution)
    _V3_SCHEDULE_CACHE.clear()
    _V3_SCHEDULE_CACHE[key] = out
    return out


def mla_attention(
        q_NTA: jax.Array,
        q_rope_TNH: jax.Array,
        k_SA: jax.Array,
        k_rope_SH: jax.Array,
        kv_cache: jax.Array,
        md: AttentionMetadata,
        mesh: Mesh,
        num_attention_heads: int,
        qk_nope_head_dim: int,
        query_nth_sharding: Sharding | None = None,
        query_tnh_sharding: Sharding | None = None,
        keyvalue_skh_sharding: Sharding | None = None,
        attn_o_nth_sharding: Sharding | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        sm_scale: float | None = None) -> Tuple[jax.Array, jax.Array]:
    """
    Main shared interface for MLA attention.  Computes the sharded attention
    output and kv cache update.

    Args:
        q_NTA: (num_query_heads, tokens_query, q_lora_rank) # head-major output from q_nope einsum projection.
        q_rope_TNH: (tokens_query, num_query_heads, head_dim)
        k_SA: (tokens_kv, q_lora_rank)
        k_rope_SH: (tokens_kv, head_dim)
        kv_cache: KV cache to be retrieved from/updated
        md: attention metadata
        mesh: Mesh
        num_attention_heads: number of attention heads
        qk_nope_head_dim: head dim for QK without rope
        query_nth_sharding: sharding to use for q_nope for the shard map (MLA kernel)
        query_tnh_sharding: sharding to use for q_rope for the shard map (MLA kernel)
        keyvalue_skh_sharding: sharding to use for k/k_rope for the shard map (MLA kernel)
        attn_o_nth_sharding: sharding to use for the attention output for the shard map (MLA kernel)
        q_scale: scale to apply to q (if quantized)
        k_scale: scale to apply to k (if quantized)
        v_scale: scale to apply to v (if quantized)
        sm_scale: softmax scale
    """
    in_specs = (
        query_nth_sharding
        or P(None, ShardingAxisName.MLP_TENSOR, None),  # q (head-major)
        query_tnh_sharding
        or P(ShardingAxisName.MLP_TENSOR, None, None),  # q_rope (token-major)
        keyvalue_skh_sharding or P(ShardingAxisName.MLP_TENSOR, None),  # k
        keyvalue_skh_sharding
        or P(ShardingAxisName.MLP_TENSOR, None),  # k_rope
        P(ShardingAxisName.BATCH),  # kv_cache
        P(ShardingAxisName.ATTN_DATA),  # md.seq_lens
        P(ShardingAxisName.ATTN_DATA),  # md.page_indices_flat
        P(ShardingAxisName.ATTN_DATA),  # md.query_start_loc
        P(ShardingAxisName.ATTN_DATA),  # md.distribution
    )
    out_specs = (
        P(ShardingAxisName.BATCH),  # kv cache
        attn_o_nth_sharding
        or P(None, ShardingAxisName.MLP_TENSOR, None)  # attn output
    )

    # v3 builds its schedule from cu_q_lens/kv_lens/page_indices/distribution,
    # none of which vary by layer -- but every layer has its own shard_map, and
    # XLA does not CSE the Pallas call, so building it inside the kernel costs
    # one `mla_metadata_schedule` launch per layer (61 on R1). Build it once for
    # the step here and hand the same pytree to every layer. The cache is keyed
    # on the metadata object, which is rebuilt each step.
    v3_schedules = None
    if mla_version.use_v3():
        v3_schedules = _v3_schedules_for_step(md, mesh, in_specs, q_NTA,
                                              q_rope_TNH, kv_cache, sm_scale,
                                              q_scale, k_scale, v_scale)
        in_specs = in_specs + (P(ShardingAxisName.ATTN_DATA),)  # schedules

    def _mla_ragged_paged_attention(q, q_rope, k, k_rope, cache, *args):
        # TODO: use auto tuner to find the best block sizes.
        num_kv_pages_per_block = _MLA_KV_PAGES_PER_BLOCK
        num_queries_per_block = _MLA_QUERIES_PER_BLOCK
        decode_batch_size = _MLA_DECODE_BATCH_SIZE

        if mla_version.use_v3():
            *args, schedules = args
            out, new_cache = mla_ragged_paged_attention_v3(
                q,
                q_rope,
                k,
                k_rope,
                cache,
                *args,
                sm_scale=sm_scale,
                schedules=schedules,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                **_v3_kernel_kwargs(cache.shape[-1]))
            return new_cache, out

        out, new_cache = mla_ragged_paged_attention(
            q,
            q_rope,
            k,
            k_rope,
            cache,
            *args,
            sm_scale=sm_scale,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            decode_batch_size=decode_batch_size,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale)

        return new_cache, out

    operands = (q_NTA, q_rope_TNH, k_SA, k_rope_SH, kv_cache, md.seq_lens,
                md.block_tables, md.query_start_loc, md.request_distribution)
    if v3_schedules is not None:
        operands = operands + (v3_schedules,)
    kv_cache, output_TNA = jax.jit(
        jax.shard_map(_mla_ragged_paged_attention,
                      mesh=mesh,
                      in_specs=in_specs,
                      out_specs=out_specs,
                      check_vma=False))(*operands)
    return kv_cache, output_TNA
