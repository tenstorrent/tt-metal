// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "barrier.hpp"
#include "ttnn/operations/ccl/barrier/device/barrier_op.hpp"

namespace ttnn::operations::ccl {

ttnn::Tensor BarrierOperation::invoke(
    const Tensor& input_tensor, const std::optional<ttnn::MemoryConfig>& memory_config, ttnn::ccl::Topology topology) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());
    return barrier_function(input_tensor, Barrier{out_memory_config, topology});
}

std::vector<ttnn::Tensor> BarrierOperation::invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensors[0].memory_config());
    std::vector<IDevice*> devices;
    devices.reserve(input_tensors.size());
    for (const auto& input_tensor : input_tensors) {
        devices.push_back(input_tensor.device());
    }
    return barrier_function(input_tensors, Barrier{out_memory_config, topology, devices});
}

}  // namespace ttnn::operations::ccl
