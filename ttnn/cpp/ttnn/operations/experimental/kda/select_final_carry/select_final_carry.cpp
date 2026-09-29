// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "select_final_carry.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "device/select_final_carry_device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor select_final_carry(
    const ttnn::Tensor& rank_final,
    const ttnn::Tensor& prefix_final,
    const ttnn::Tensor& actual_start,
    uint32_t local_rows,
    const std::optional<ttnn::Tensor>& actual_end,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    uint32_t sequence_parallel_axis,
    std::optional<uint32_t> num_links) {
    TT_FATAL(
        rank_final.storage_type() == StorageType::DEVICE && rank_final.buffer() != nullptr,
        "select_final_carry: rank_final must be an allocated device tensor");
    TT_FATAL(
        prefix_final.storage_type() == StorageType::DEVICE && prefix_final.buffer() != nullptr,
        "select_final_carry: prefix_final must be an allocated device tensor");
    auto* mesh = prefix_final.device();
    const auto topology = ::ttnn::ccl::convert_2d_to_1d_topology(
        ::ttnn::ccl::get_usable_topology(prefix_final, tt::tt_fabric::get_fabric_topology(), sequence_parallel_axis));
    return ttnn::experimental::prim::select_final_carry(
        rank_final,
        prefix_final,
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        actual_start,
        actual_end,
        sequence_parallel_axis,
        local_rows,
        num_links.value_or(
            static_cast<uint32_t>(ttnn::operations::ccl::common::get_num_links(*mesh, sequence_parallel_axis))),
        topology);
}

}  // namespace ttnn::experimental::kda
