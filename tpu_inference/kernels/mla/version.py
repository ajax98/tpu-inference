# SPDX-License-Identifier: Apache-2.0
"""Which MLA kernel to run, selected by the MLA_VERSION environment variable.

The KV cache layout differs between versions, so the choice has to be made
identically in two places that do not otherwise talk to each other:
`runner/kv_cache.py` (which allocates the cache) and
`layers/common/attention_interface.py` (which reads it). Both go through here.

  v2 (default) -- tokens on sublanes, `[pages, page_size // packing, packing, kv_dim]`
  v3           -- tokens on lanes,    `[pages, kv_dim, page_size]`
  v3x          -- v3 plus the XLA-side experiments in `v3_xla_fixes`
"""

import os


def _raw() -> str:
  v = os.getenv("MLA_VERSION", "v2").strip().lower()
  if v not in ("v2", "v3", "v3x"):
    raise ValueError(f"MLA_VERSION must be v2, v3 or v3x, got {v!r}")
  return v


def get_mla_version() -> str:
  """Returns "v2" or "v3" (v3x runs the v3 kernel and cache layout)."""
  return "v3" if _raw() == "v3x" else _raw()


def v3_xla_fixes() -> bool:
  """v3x: pin new-KV layout and cap the kernel's VMEM claim."""
  return _raw() == "v3x"


def use_v3() -> bool:
  return get_mla_version() == "v3"
