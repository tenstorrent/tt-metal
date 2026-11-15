// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sample_candidates.hpp"

#include "device/sample_candidates_device_operation.hpp"

namespace ttnn::experimental::deepseek {

std::vector<Tensor> decode_sample_candidates(
    const Tensor& logits,
    const Tensor& inv_temp,
    const Tensor& index_offset,
    uint32_t k,
    float fill_value,
    uint32_t num_workers) {
    return ttnn::prim::decode_sample_candidates(logits, inv_temp, index_offset, k, fill_value, num_workers);
}

}  // namespace ttnn::experimental::deepseek
