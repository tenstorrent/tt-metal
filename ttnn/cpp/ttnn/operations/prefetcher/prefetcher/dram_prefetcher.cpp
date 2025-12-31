// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher.hpp"
#include <optional>
#include <tt-metalium/global_circular_buffer.hpp>
#include "device/dram_prefetcher_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::dram_prefetcher {

Tensor dram_prefetcher(
    std::vector<ttnn::Tensor>& tensors,
    const uint32_t num_layers,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    const bool enable_performance_mode) {
    using OperationType = DramPrefetcherOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .num_layers = num_layers,
        .enable_performance_mode = enable_performance_mode,
        .global_cb = global_cb,
    };
    auto tensor_args = OperationType::tensor_args_t{.input_tensors = tensors};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::operations::dram_prefetcher
