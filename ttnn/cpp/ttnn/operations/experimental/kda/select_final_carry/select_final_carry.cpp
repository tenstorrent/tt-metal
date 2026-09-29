// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "select_final_carry.hpp"

#include "device/select_final_carry_device_operation.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor select_final_carry(
    const ttnn::Tensor& rank_finals,
    const ttnn::Tensor& prefix_final,
    const ttnn::Tensor& actual_start,
    uint32_t local_rows,
    const std::optional<ttnn::Tensor>& actual_end,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    uint32_t sequence_parallel_axis) {
    TT_FATAL(
        rank_finals.storage_type() == StorageType::DEVICE && rank_finals.buffer() != nullptr,
        "select_final_carry: rank_finals must be an allocated device tensor");
    TT_FATAL(
        prefix_final.storage_type() == StorageType::DEVICE && prefix_final.buffer() != nullptr,
        "select_final_carry: prefix_final must be an allocated device tensor");
    return ttnn::experimental::prim::select_final_carry(
        rank_finals,
        prefix_final,
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        actual_start,
        actual_end,
        sequence_parallel_axis,
        local_rows);
}

}  // namespace ttnn::experimental::kda
