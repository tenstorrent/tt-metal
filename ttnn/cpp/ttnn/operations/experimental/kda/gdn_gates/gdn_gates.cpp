// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gdn_gates.hpp"

#include "device/gdn_gates_device_operation.hpp"

namespace ttnn::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor> gdn_gates(
    const ttnn::Tensor& gab,
    const ttnn::Tensor& dt_bias,
    const ttnn::Tensor& a_neg,
    uint32_t a_col_offset,
    uint32_t b_col_offset,
    uint32_t num_heads,
    float beta_scale,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    TT_FATAL(
        gab.storage_type() == StorageType::DEVICE && gab.buffer() != nullptr,
        "gdn_gates: gab must be an allocated device tensor");
    const auto out_mc = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    return ttnn::experimental::prim::gdn_gates(
        gab, dt_bias, a_neg, a_col_offset, b_col_offset, num_heads, beta_scale, out_mc);
}

}  // namespace ttnn::experimental
