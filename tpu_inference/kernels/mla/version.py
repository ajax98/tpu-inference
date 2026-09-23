# SPDX-License-Identifier: Apache-2.0
"""Which MLA kernel to run, selected by the MLA_VERSION environment variable.

The KV cache layout differs between versions, so the choice has to be made
identically in two places that do not otherwise talk to each other:
`runner/kv_cache.py` (which allocates the cache) and
`layers/common/attention_interface.py` (which reads it). Both go through here.

  v2 (default) -- tokens on sublanes, `[pages, page_size // packing, packing, kv_dim]`
  v3           -- tokens on lanes,    `[pages, kv_dim, page_size]`
"""

import os


def get_mla_version() -> str:
  """Returns "v2" or "v3"."""
  v = os.getenv("MLA_VERSION", "v2").strip().lower()
  if v not in ("v2", "v3"):
    raise ValueError(f"MLA_VERSION must be v2 or v3, got {v!r}")
  return v


def use_v3() -> bool:
  return get_mla_version() == "v3"
