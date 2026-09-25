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

"""Batched Multi-Head Latent Attention (MLA) v3 for TPU."""

from tpu_inference.kernels.mla.v3 import bref_override
from tpu_inference.kernels.mla.v3 import configs
from tpu_inference.kernels.mla.v3 import flash_attention
from tpu_inference.kernels.mla.v3 import kernel
from tpu_inference.kernels.mla.v3 import mla_wrapper
from tpu_inference.kernels.mla.v3 import schedule
from tpu_inference.kernels.mla.v3 import utils

BlockSizes = configs.BlockSizes
MlaModelConfigs = configs.MlaModelConfigs
ServingConfigs = configs.ServingConfigs
KVLayout = configs.KVLayout
MlaCase = configs.MlaCase
MlaConfigs = configs.MlaConfigs

MlaSchedule = schedule.MlaSchedule
generate_mla_metadata = schedule.generate_mla_metadata

flash_attention_qk_softmax = flash_attention.flash_attention_qk_softmax
flash_attention_pv = flash_attention.flash_attention_pv
chunked_flash_attention = flash_attention.chunked_flash_attention

KVBufferedRefSeqAlongLane = bref_override.KVBufferedRefSeqAlongLane
# One combined query ref replaces the separate nope/pe pair. The upstream
# branch renamed the class but left these two exports pointing at the old
# names, so importing the package raised AttributeError.
BatchingQRef = bref_override.BatchingQRef
BatchingORef = bref_override.BatchingORef

mla_ragged_paged_attention = kernel.mla_ragged_paged_attention
static_validate_inputs = kernel.static_validate_inputs
prepare_q_nope_inputs = kernel.prepare_q_nope_inputs
prepare_q_inputs = kernel.prepare_q_inputs
prepare_kv_inputs_for_transposed_kv_cache = kernel.prepare_kv_inputs_for_transposed_kv_cache
prepare_outputs = kernel.prepare_outputs
transpose_kv_cache_to_v3 = utils.transpose_kv_cache_to_v3
transpose_kv_cache_from_v3 = utils.transpose_kv_cache_from_v3

