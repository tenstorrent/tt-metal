// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "exchange_histories.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "device/exchange_histories_device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::experimental::kda {

std::vector<ttnn::Tensor> exchange_histories(
    const ttnn::Tensor& projected,
    uint32_t width,
    const ttnn::Tensor& actual_start,
    uint32_t local_rows,
    const std::optional<ttnn::Tensor>& actual_end,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    uint32_t sequence_parallel_axis) {
    TT_FATAL(
        projected.storage_type() == StorageType::DEVICE && projected.buffer() != nullptr,
        "exchange_histories: projected must be an allocated device tensor");
    const auto topology = ::ttnn::ccl::convert_2d_to_1d_topology(
        ::ttnn::ccl::get_usable_topology(projected, tt::tt_fabric::get_fabric_topology(), sequence_parallel_axis));
    return ttnn::experimental::prim::exchange_histories(
        projected,
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        actual_start,
        actual_end,
        sequence_parallel_axis,
        local_rows,
        width,
        topology);
}

}  // namespace ttnn::experimental::kda
