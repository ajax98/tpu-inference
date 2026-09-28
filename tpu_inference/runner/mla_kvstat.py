"""Per-step, per-DP-rank decode KV-length stats for the MLA v2/v3 comparison.

Host-side only: reads the per-rank seq_lens the runner already built for the
attention kernel, so the device program is unchanged. One line per step goes to
the log, prefixed MLA_KVSTAT, as JSON:

  {"s": step, "t": unix time, "pad": padded tokens per rank, "r": [rank, ...],
   "kv": [[decode kv_len, ...] per rank]}

with each rank in "r" a list in the order of FIELDS, and "kv" the raw decode
kv_lens so the block counts can be recomputed offline for any bkv or group.
Decodes are the first n_dec entries of each rank's seq_lens (the batch is
reordered decode-first), which is also the order v2 groups them in.

Decode work is counted in the decode pass's KV blocks (DECODE_BKV tokens):
  v2_bd   sum over full groups of 4 of cdiv(max kv in group, bkv) -- MLA-bd
          loops every group to its longest member
  v2_d    sum over the n_dec % 4 leftovers of cdiv(kv, bkv) -- MLA-d runs them
          one at a time, one block per step
  tiles   sum over all decodes of cdiv(kv, bkv) -- the work itself
  v3      cdiv(tiles, 4) -- v3's schedule packs tiles 4 per step across
          sequences

Every SUMMARY_EVERY steps a cumulative MLA_KVSTAT_SUM line is logged too.
Set MLA_KVSTAT=0 to disable.
"""
import json
import os
import time

import numpy as np

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("MLA_KVSTAT", "1") != "0"
DECODE_BKV = 3 * 1024  # _MLA_KV_PAGES_PER_BLOCK[0] pages of 1024 tokens
GROUP = 4  # _MLA_DECODE_BATCH_SIZE
SUMMARY_EVERY = 1000
FIELDS = ("n_dec", "n_pref", "pref_tok", "kv_min", "kv_p25", "kv_p50",
          "kv_p75", "kv_max", "groups", "rem", "v2_bd", "v2_d", "tiles", "v3")


def rank_stats(dec_kv: np.ndarray, n_pref: int, pref_tok: int) -> list[int]:
    n = len(dec_kv)
    if n == 0:
        return [0, n_pref, pref_tok] + [0] * (len(FIELDS) - 3)
    blocks = -(-dec_kv // DECODE_BKV)
    g = n // GROUP
    v2_bd = int(blocks[:g * GROUP].reshape(g, GROUP).max(axis=1).sum()) if g else 0
    v2_d = int(blocks[g * GROUP:].sum())
    tiles = int(blocks.sum())
    q = np.percentile(dec_kv, [0, 25, 50, 75, 100]).astype(int).tolist()
    return [n, n_pref, pref_tok, *q, g, n % GROUP, v2_bd, v2_d, tiles,
            -(-tiles // GROUP)]


class KvStat:

    def __init__(self):
        self.step = 0
        self.tot = dict(steps=0, dec_steps=0, v2_bd=0, v2_d=0, tiles=0, v3=0,
                        rem_hist=[0] * GROUP)
        logger.info("MLA_KVSTAT_FIELDS %s",
                    json.dumps(dict(fields=FIELDS, bkv=DECODE_BKV, group=GROUP)))

    def record(self, padded_tokens_per_rank: int, ranks: list[list[int]],
               kv: list[list[int]]):
        self.step += 1
        t = self.tot
        t["steps"] += 1
        if any(r[0] for r in ranks):
            t["dec_steps"] += 1
        for r in ranks:
            if r[0]:
                t["rem_hist"][r[9]] += 1
            t["v2_bd"] += r[10]
            t["v2_d"] += r[11]
            t["tiles"] += r[12]
            t["v3"] += r[13]
        logger.info("MLA_KVSTAT %s", json.dumps(
            dict(s=self.step, t=round(time.time(), 3),
                 pad=int(padded_tokens_per_rank), r=ranks, kv=kv),
            separators=(",", ":")))
        if self.step % SUMMARY_EVERY == 0:
            logger.info("MLA_KVSTAT_SUM %s", json.dumps(dict(s=self.step, **t)))
