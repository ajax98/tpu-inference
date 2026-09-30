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
