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

"""Configuration dataclasses, dimensions, and validation for Batched MLA.

This module defines configuration structures and hardware layout rules for
Multi-Head Latent Attention (MLA) on TPU.
"""

import dataclasses
import enum
from typing import Any

from jax.experimental import pallas as pl
import jax.numpy as jnp
from tpu_inference.kernels.mla.v3 import utils


@dataclasses.dataclass(frozen=True)
class BlockSizes:
  """Tuning block sizes and tiling parameters for the MLA kernel.

  Attributes:
    bq_sz: Query block size (number of sequence tokens per Q block).
    bq_c_sz: Chunked query/compute block size for split execution.
    bkv_sz: KV cache block size (number of context tokens processed per step).
    batch_size: Number of batch lanes packed into one kernel step. The lanes are
      chained *serially* within a step -- lane i's online-softmax state (m, l,
      acc) is rolled forward into lane i+1 -- so this is a work-per-step /
      DMA-overlap knob, not a parallelism knob.
    n_buffer: Pipelining buffer depth (e.g. 2 for double buffering).
  """

  bq_sz: int
  bq_c_sz: int
  bkv_sz: int
  batch_size: int
  n_buffer: int


@dataclasses.dataclass(frozen=True)
class MlaModelConfigs:
  """Immutable architectural parameters of the MLA model.

  MLA decomposes attention representations into:
    - Non-positional latent query/key/value vectors of dimension `lkv_dim`
      (e.g., 512 in DeepSeek-V2/V3).
    - Decoupled rotary embedding (RoPE) query/key vectors of dimension `r_dim`
      (e.g., 64 in DeepSeek-V2/V3).
    - Single compressed KV head (H_kv = 1) shared across all `num_q_heads`
    (e.g., 128).

  Attributes:
    num_q_heads: Number of query attention heads (e.g. 128).
    lkv_dim: Dimension of latent non-positional KV vector d_nope (e.g. 512).
    r_dim: Dimension of decoupled RoPE positional key vector d_pe (e.g. 64).
    mask_value: Large negative value used for causal/padding masking.
    sm_scale: Softmax temperature scale factor (default: 1.0 / sqrt(d_q)).
    soft_cap: Optional logit soft-capping threshold (e.g., Gemma-2 style).
    sliding_window: Optional sliding window attention horizon in tokens.
  """

  num_q_heads: int
  lkv_dim: int
  r_dim: int
  mask_value: float
  sm_scale: float = 1.0
  soft_cap: float | None = None
  sliding_window: int | None = None


class KVLayout(enum.StrEnum):
  """Memory layout of the paged KV cache in HBM and VMEM.

  - SEQ_ALONG_LANE: Sequence tokens are packed along the 128 TPU physical lanes;
      latent dimension is on sublanes. Cache is [pages, kv_dim, page_size].

  A HEAD_ALONG_LANE variant (v2's orientation: tokens on sublanes,
  [pages, page_size/packing, packing, kv_dim]) was implemented and measured. It
  removes the new-token merge entirely -- tokens can be DMA'd to their logical
  offset because the token axis is untiled -- but it was 8.9% slower at 8
  sequences and 21.4% slower at 32, against a within-config sd of ~1.5 us. The
  merge was never the cost; the SEQ orientation is simply the better operand
  layout for the attention matmuls. Removed rather than kept as dead config.
  """

  SEQ_ALONG_LANE = enum.auto()

  @property
  def symbol(self) -> str:
    return "snh"


@dataclasses.dataclass(frozen=True)
class ServingConfigs:
  """Workload and serving batch configuration parameters.

  Attributes:
    num_seqs: Maximum number of active sequences in the current batch.
    page_size: Paged memory page size in tokens (must be multiple of 128 for
      SEQ_ALONG_LANE).
    total_q_tokens: Total flattened query tokens across all sequences (sum of
      q_lens).
    num_page_indices: Size of the flattened page index table (num_seqs *
      pages_per_seq).
    dtype_q: Data type for query tensors (e.g., jnp.bfloat16).
    dtype_kv: Data type for KV cache tensors (e.g., jnp.bfloat16).
    dtype_out: Data type for attention output tensors (e.g., jnp.bfloat16).
    scale_q: Optional scalar quantization multiplier for Q.
    scale_k: Optional scalar quantization multiplier for K.
    scale_v: Optional scalar quantization multiplier for V.
    kv_layout: Paged memory layout. Only SEQ_ALONG_LANE is implemented.
    smem_fraction_limit_for_schedule_generation: SMEM budget limit fraction.
    max_schedule_size_multiplier: Floor on the multiplier sizing the HBM
      schedule, in units of `MlaConfigs.max_steps_ub`. Raising it only
      over-allocates; `MlaConfigs.max_schedule_size_multiplier` already raises
      it on its own for any shape that provably needs more.
  """

  num_seqs: int
  page_size: int
  total_q_tokens: int
  num_page_indices: int
  dtype_q: jnp.dtype
  dtype_kv: jnp.dtype
  dtype_out: jnp.dtype
  scale_q: float | None = None
  scale_k: float | None = None
  scale_v: float | None = None
  kv_layout: KVLayout = KVLayout.SEQ_ALONG_LANE
  smem_fraction_limit_for_schedule_generation: float = 0.33
  max_schedule_size_multiplier: int = 16

  # --- Tuning flags -------------------------------------------------------
  #
  # What remains here are the flags whose best value depends on the shape. The
  # ones that measured as wins everywhere are gone: their behaviour is now
  # unconditional in the kernel rather than being a flag defaulted to True.
  # See the optimization log for the per-spec numbers behind each removal.
  #
  # Reference point, `decode_f8_kv9216` at page_size=1024, device kernel time
  # (TPU7x, jax 0.11.1): baseline 0.6567 ms, v2 0.4033 ms.
  #
  #     kv_slack_pad_lanes=128   -10.5%    decode only; 0 on whole-prompt
  #                                        prefill, see the field comment
  #     s_dtype=bf16              +1.9%    HURTS on decode
  #     p_same_dtype_as_v         +5.9%    HURTS on decode, -16.1% on
  #                                        chunked_prefill_f8_kv8192

  # --- Port of v2's `p_same_dtype_as_v`. HURTS ON DECODE but flips sign on
  # --- prefill, which is why it stays configurable.
  #
  # This was the leading explanation for v3's kernel gap -- v2 has it and its
  # autotuner selected it on every workload. Porting it faithfully made v3
  # *slower* on decode, which is what established that v3's decode bottleneck
  # is vector-unit and VMEM access work, not MXU operand width.
  #
  # Narrowing the QK scores to bf16 before masking and softmax was also tried
  # and removed: MEASURED +1.9%, because the `astype` is itself a full-tile
  # vector pass and at bq_sz=1 the tile is too small for the saving to pay.
  # p_same_dtype_as_v: cast softmax probabilities to the KV dtype before the PV
  #   matmul, so the MXU sees fp8 x fp8 rather than f32 x fp8. Operand width,
  #   not accumulator width -- accumulation stays f32 either way.
  #   MEASURED +5.9% on decode, -16.1% on chunked_prefill_f8_kv8192. The
  #   mechanism is unconfirmed -- see the note in `v3_op.Config` and
  #   bench_pv_operand_dtypes.py.
  p_same_dtype_as_v: bool = False


  # kv_slack_pad_lanes: extra lanes appended to the KV staging buffer purely to
  #   change its *stride*, not its capacity.
  #
  #   `_stitch_decode_lane` and `store_new_kv_lane` walk the buffer with
  #   `pl.ds(start, outer_dim, lanes_per_col)` where
  #   `lanes_per_col = kv_vmem_lanes // 128`. That stride lands on VMEM banks,
  #   and a power-of-two stride aliases every access onto the same bank.
  #
  #   MEASURED -10.5%, the largest single win found. Four-point sweep, kernel
  #   time, everything else held at the best config:
  #
  #       v_len   lanes_per_col   parity   kernel
  #        5248        41          odd     0.5454   <- default slack + 128
  #        4224        33          odd     0.5770
  #        5120        40          even    0.6093   <- default, no pad
  #        4096        32          even    0.9435   <- tight slack, no pad
  #
  #   Both odd strides beat both even ones, and the pure power of two is
  #   catastrophic. 4224 loses to 5248 despite being 20% smaller, so capacity
  #   is not the variable.
  #
  #   128 is correct for *this* bkv_sz and page_size, not universally: the
  #   requirement is that `kv_vmem_lanes // 128` come out odd. Re-derive when
  #   either changes.
  #
  #   Corollary: the original `+2 * page_size` slack is load-bearing by
  #   accident. It is not waste, and trimming it to any power of two is a trap.
  kv_slack_pad_lanes: int = 0









  @property
  def pages_per_seq(self) -> int:
    return self.num_page_indices // self.num_seqs

  @property
  def page_size_log2(self) -> int:
    return (self.page_size - 1).bit_length()

  @property
  def packing_q(self) -> int:
    """Number of elements packed per 32-bit word (e.g. 2 for bfloat16)."""
    return utils.get_dtype_packing(self.dtype_q)

  @property
  def packing_kv(self) -> int:
    """Number of elements packed per 32-bit word (e.g. 2 for bfloat16)."""
    return utils.get_dtype_packing(self.dtype_kv)


class MlaCase(enum.StrEnum):
  """Execution mode for the MLA kernel.

  - DECODE: All sequences are in autoregressive decode (q_len = 1).
  - PREFILL: All sequences are in prompt prefill (q_len > 1, static).
  - MIXED: Batch contains a combination of prefill and decode sequences.
  """

  DECODE = enum.auto()
  PREFILL = enum.auto()
  MIXED = enum.auto()

  @property
  def symbol(self) -> str:
    match self:
      case MlaCase.DECODE:
        return "d"
      case MlaCase.PREFILL:
        return "p"
      case MlaCase.MIXED:
        return "m"

  def get_range(self, distribution: Any) -> tuple[Any, Any]:
    """Extracts sequence start and end indices for this execution mode."""
    match self:
      case MlaCase.DECODE:
        return 0, distribution[0]
      case MlaCase.PREFILL:
        return distribution[0], distribution[1]
      case MlaCase.MIXED:
        return distribution[1], distribution[2]


@dataclasses.dataclass(frozen=True, eq=True)
class MlaConfigs:
  """Master configuration combining block sizes, model, serving, and hardware constraints."""

  block: BlockSizes
  model: MlaModelConfigs
  serve: ServingConfigs
  mode: MlaCase
  vmem_limit_bytes: int = 16 * 1024 * 1024

  # Expose block sizes directly for convenient access
  @property
  def bq_sz(self) -> int:
    return self.block.bq_sz

  @property
  def bq_c_sz(self) -> int:
    return self.block.bq_c_sz

  @property
  def bkv_sz(self) -> int:
    return self.block.bkv_sz

  @property
  def batch_size(self) -> int:
    return self.block.batch_size

  @property
  def n_buffer(self) -> int:
    return self.block.n_buffer

  @property
  def q_split(self) -> int:
    """Number of query sub-chunks (bq_sz // bq_c_sz)."""
    return max(1, self.bq_sz // self.bq_c_sz)

  # Derived hardware alignment dimensions
  @property
  def kv_dim_align(self) -> int:
    """Granularity the KV sub-dimensions are padded to in VMEM: 128 lanes.

    This is the compute view and must stay lane-aligned. Shrinking it to
    `packing_kv * 8` also shrinks `aligned_r_dim` -- and `q_pe` is reshaped
    with that as its **minor** dimension. A TPU vector's minor dim is the lane
    axis, so Mosaic rejects the result:

        tpu.reshape : (vector<1x2x1x16x4x128xf8E4M3FN>) -> vector<2x128x64>
        infer-vector-layout: unsupported shape cast

    """
    return utils.get_tpu_num_lanes()

  @property
  def aligned_lkv_dim(self) -> int:
    """d_nope (512), padded to 128 lanes."""
    return utils.align_to(self.model.lkv_dim, self.kv_dim_align)

  @property
  def aligned_r_dim(self) -> int:
    """d_pe (64 -> 128), padded to 128 lanes. See `kv_dim_align`."""
    return utils.align_to(self.model.r_dim, self.kv_dim_align)




  @property
  def lkv_sublanes(self) -> int:
    """Sublanes the latent part occupies. Same in HBM and VMEM."""
    return self.aligned_lkv_dim // self.serve.packing_kv

  @property
  def aligned_kv_dim(self) -> int:
    """Combined KV dimension [C_kv, K_pe] aligned to 128-byte multiples."""
    return self.aligned_lkv_dim + self.aligned_r_dim

  @property
  def aligned_q_dim(self) -> int:
    """Combined Query dimension [Q_nope, Q_pe] aligned to 128-byte multiples."""
    return self.aligned_lkv_dim + self.aligned_r_dim

  @property
  def aligned_num_q_heads(self) -> int:
    """Number of Q heads aligned to word-packing boundary."""
    packing_q = self.serve.packing_q
    return utils.align_to(self.model.num_q_heads, packing_q)

  # Paged KV block calculations
  @property
  def bkv_p(self) -> int:
    """Base number of physical pages spanning a single KV block."""
    return pl.cdiv(self.block.bkv_sz, self.serve.page_size)

  @property
  def bkv_p_cache(self) -> int:
    """Number of pages to fetch from the existing cached KV table per step.

    At most bkv_p pages, since cached pages are already pre-sliced at offset 0.

    This used to return 0 in PREFILL mode on the assumption that a prefill
    sequence has no history. That only holds for a *whole-prompt* prefill; under
    chunked prefill the earlier chunks are already in the paged cache
    (kv_len > q_len) and skipping the cache fetch silently drops them from the
    attention. PREFILL therefore fetches the cache exactly like MIXED.
    """
    return self.bkv_p

  @property
  def one_new_token(self) -> bool:
    """Whether a lane-task can receive at most one new (unpaged) KV token.

    This is a statement about the *sequence*, not about the query block, and
    that distinction is a real correctness boundary. `bq_sz == 1` was used for
    it historically, which is wrong: a prefill sequence processed one query
    token at a time still has `q_len` new KV tokens to stitch in. On
    `[(256, 1024)]` at `num_queries_per_block=1` that took the decode stitch
    path, which writes exactly one lane and *zeroes* everything past the
    boundary -- so 255 of 256 new tokens were dropped. Query token `i` attends
    to `i` of those zeroed positions, so the error grew with `i` and crossed
    tolerance around `i = 20`, mismatching 1.372% of the output.

    Only DECODE guarantees one new token per sequence, so only DECODE may take
    the fast paths gated on this.
    """
    return self.mode == MlaCase.DECODE

  @property
  def bkv_p_new(self) -> int:
    """Number of pages to fetch from the unpaged new tokens tensor per step.

    - In DECODE (bq_sz = 1): Exactly 1 new token is decoded, spanning at most 1
    page.
    - Otherwise: unaligned sequence starts in unpaged HBM can straddle across an
      extra page boundary, requiring (bkv_p + 1) page fetches.
    """
    if self.one_new_token:
      return 1
    return self.bkv_p + 1

  @property
  def dma_kv_new_size(self) -> int:
    """Number of int32 descriptor fields per new-token DMA struct entry.

    Five, matching `schedule.SeqAlongLaneDmaNew`: fetch_hbm, fetch_vmem, wb_hbm,
    wb_vmem, and the packed flags word.
    """
    return 5

  @property
  def fuse_accum(self) -> bool:
    """Whether to unconditionally normalize acc/l on every block to avoid jax.lax.cond.

    In DECODE (bq_sz = 1), vector division acc / l takes negligible ALU cycles,
    so running it unconditionally eliminates the compiler scheduling barrier.
    In PREFILL (bq_sz >= 64), dividing full matrices is heavy, so we
    conditionally
    execute normalization only on the last block (fuse_accum = False).

    Measured: forcing the conditional form on DECODE costs 2.4%, so the
    docstring's reasoning holds -- the `lax.cond` scheduling barrier is worth
    more than the normalize/store it elides.
    """
    return self.mode == MlaCase.DECODE

  # Per-token byte sizes for DMA lane wait synchronization
  @property
  def kv_bytes_per_token(self) -> int:
    """Byte count transferred per KV token (1 shared latent stream).

    Must be the **HBM** width, not the VMEM width. `_compute_waits` turns this
    into `total_wait_kv_in`, and `KVBufferedRefSeqAlongLane.wait_in` blocks
    until that many bytes have landed, so it must match what the DMA actually
    moves. The HBM cache and the VMEM staging buffer are the same width, so
    that is simply `aligned_kv_dim`.
    """
    return self.aligned_kv_dim * jnp.dtype(self.serve.dtype_kv).itemsize

  @property
  def q_bytes_per_token(self) -> int:
    """Byte count transferred per Query token (H_q heads * (d_nope + d_pe))."""
    return (
        self.aligned_num_q_heads
        * self.aligned_q_dim
        * jnp.dtype(self.serve.dtype_q).itemsize
    )

  @property
  def o_bytes_per_token(self) -> int:
    """Byte count transferred per Output token (H_q heads * d_nope).

    Note: RoPE key is excluded from output; output width is strictly d_nope
    (512).
    """
    return (
        self.aligned_num_q_heads
        * self.aligned_lkv_dim
        * jnp.dtype(self.serve.dtype_out).itemsize
    )

  @property
  def max_steps_ub(self) -> int:
    """Calculates the maximum schedule steps that can fit in TPU SMEM."""
    fixed_bytes = (
        self.serve.num_seqs  # kv_lens
        + (self.serve.num_seqs + 1)  # cu_q_lens
        + (self.serve.num_seqs * self.serve.pages_per_seq)  # page_indices
        + 3  # distribution [decode, prefill, total]
        + self.block.batch_size  # lane_lengths
        + 1  # actual_steps
    ) * 4  # 4 bytes per int32

    smem_limit_bytes = (
        utils.get_tpu_smem_capacity_bytes() - 32 * 1024
    ) * self.serve.smem_fraction_limit_for_schedule_generation
    available_bytes = smem_limit_bytes - fixed_bytes

    bytes_scalars_per_lane = 28
    bytes_cache_dma_per_lane = 12 * self.bkv_p_cache
    bytes_new_dma_per_lane = 4 * self.dma_kv_new_size * self.bkv_p_new
    bytes_global_waits = 16

    bytes_per_step = (
        bytes_scalars_per_lane
        + bytes_cache_dma_per_lane
        + bytes_new_dma_per_lane
    ) * self.block.batch_size + bytes_global_waits

    max_steps_ub = available_bytes // bytes_per_step
    num_lanes = utils.get_tpu_num_lanes()
    return int(max(1, max_steps_ub // num_lanes) * num_lanes)

  # Scratch buffer shapes in VMEM
  @property
  def lm_scratch_shape(self) -> tuple[int, ...]:
    """Scratch shape for running row-max (m) and row-sum (l) vectors."""
    num_lanes = utils.get_tpu_num_lanes()
    return (
        self.block.bq_sz * self.aligned_num_q_heads,
        num_lanes,
    )

  @property
  def acc_scratch_shape(self) -> tuple[int, ...]:
    """Scratch shape for accumulator matrix (width is strictly d_nope = 512)."""
    return (
        self.block.bq_sz * self.aligned_num_q_heads,
        self.aligned_lkv_dim,
    )

  @property
  def kv_vmem_lanes(self) -> int:
    """Lane extent of the KV staging buffer: `bkv_sz` plus stitch slack.

    The slack holds new-KV pages fetched *past* the cached region before
    `stitch_*_lane` rolls them into place. `fill_dma_kv_new` writes
    `page_size` bytes at `fetch_vmem = (cache_pages + i) * page_size`, so the
    buffer must cover `(cache_pages + i + 1) * page_size`. Two pages in
    general: one for rounding `bkv_sz_cache` up to a page, one for the new
    tokens' own intra-page offset.

    **Decode needs only one.** With `q_len == 1`,
    `new_sz = min(bkv_sz - bkv_sz_cache, kv_left_frm_new) <= 1` token, so
    `num_pages_to_fetch == 1` and only `i = 0` runs -- which the `bq_sz == 1`
    branch in `schedule.fill_dma_kv_new` already asserts via
    `bkv_p_new == 1`. Then `fetch_vmem = cache_pages * page_size <= bkv_sz`
    and the write extends one page: `bkv_sz + page_size` suffices, and the
    second page is never touched.

    Reserving it anyway costs 20% of the buffer at page_size=1024 and widens
    the DMA destination stride (5120 vs 4096 lanes for a 1024-byte write),
    which is why this is worth a flag rather than left as a constant.

    The guard mirrors the condition `stitch_new_kv_lane` uses to select its
    O(1) path, so it cannot apply to a multi-token query block.
    """
    # The second page is dead weight on decode, exactly as described above,
    # and it is not free: at bkv=3/page=1024/batch=8 it is 10.5 MB of a 64 MB
    # VMEM, which is what put `batch=8` 1.55 MB over the limit and forced
    # batch=4 -- doubling the grid-step count that v3's remaining decode
    # deficit is proportional to.
    slack_pages = 1 if self.one_new_token else 2
    return (
        self.block.bkv_sz
        + slack_pages * self.serve.page_size
        + self.serve.kv_slack_pad_lanes
    )

  @property
  def kv_vmem_shape(self) -> tuple[int, ...]:
    """VMEM allocation shape for the KV staging buffer.

    [batch, aligned_kv_dim, tokens] -- tokens minormost, i.e. on lanes.
    """
    return (self.block.batch_size, self.aligned_kv_dim, self.kv_vmem_lanes)

  @property
  def q_vmem_shape(self) -> tuple[int, ...]:
    """VMEM allocation shape for combined Query buffer [batch_size, bq_sz, num_q_heads, aligned_q_dim]."""
    return (
        self.block.batch_size,
        self.block.bq_sz,
        self.aligned_num_q_heads,
        self.aligned_q_dim,
    )

  @property
  def o_vmem_shape(self) -> tuple[int, ...]:
    """VMEM allocation shape for output buffer [batch_size, bq_sz, num_q_heads, lkv_dim]."""
    return (
        self.block.batch_size,
        self.block.bq_sz,
        self.aligned_num_q_heads,
        self.aligned_lkv_dim,
    )

  @property
  def max_steps_needed(self) -> int:
    """Upper bound on the schedule steps *any* input of this shape can need.

    The schedule loop increments a counter once per (sequence, q-block, k-block)
    task and packs `batch_size` consecutive tasks into one step, so bounding the
    task count bounds the step count. Both factors below hold for every ragged
    split of `total_q_tokens` across `num_seqs` sequences, which is what makes
    this computable at trace time from shapes alone.

    Nothing here can be tightened by looking at the actual `cu_q_lens` /
    `kv_lens`, because those are runtime values; the bound has to cover the
    worst split the shapes permit. It does assume `kv_len <= pages_per_seq *
    page_size` for every sequence, but a `kv_len` past the end of its page table
    is already an out-of-contract input that the kernel would read the wrong
    pages for.
    """
    # Sequence `s` contributes `cdiv(q_len_s, bq_sz)` q-blocks. Two bounds on
    # the sum are available and neither dominates: `cdiv(q, b) <= q` is tight
    # for decode (many one-token sequences), `cdiv(q, b) <= q // b + 1` is tight
    # for prefill (few long ones).
    q_blocks_ub = min(
        self.serve.total_q_tokens,
        self.serve.num_seqs + self.serve.total_q_tokens // self.block.bq_sz,
    )
    # A q-block visits at most every KV block the page table can address for
    # its sequence. Causal and sliding-window masking only remove blocks from
    # that range, so ignoring both is safe (and, for prefill, loose by ~2x).
    k_blocks_ub = pl.cdiv(
        self.serve.pages_per_seq * self.serve.page_size, self.block.bkv_sz
    )
    return pl.cdiv(q_blocks_ub * k_blocks_ub, self.block.batch_size)

  @property
  def max_schedule_size_multiplier(self) -> int:
    """Sizes the HBM schedule at `max_steps_ub *` this, in steps.

    Raised above the configured value whenever the shape provably needs more
    room, which is what keeps the overflow guard in `_write_schedule_to_hbm`
    from ever firing. Sizing is the fix rather than validation: the bound is a
    worst-case over ragged splits, so rejecting shapes that merely *might*
    overflow would reject shapes that in practice never do, whereas
    over-allocating costs only HBM - a few hundred bytes per step, in a buffer
    the kernel streams rather than resides in.
    """
    return max(
        self.serve.max_schedule_size_multiplier,
        pl.cdiv(self.max_steps_needed, self.max_steps_ub),
    )