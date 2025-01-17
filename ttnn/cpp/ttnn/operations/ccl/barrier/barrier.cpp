// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "barrier.hpp"
#include "cpp/ttnn/operations/ccl/barrier/device/barrier_op.hpp"

namespace ttnn::operations::ccl {

ttnn::Tensor BarrierOperation::invoke(
    const Tensor& input_tensor, const std::optional<ttnn::MemoryConfig>& memory_config, ttnn::ccl::Topology topology) {
    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());
    return barrier_function(input_tensor, Barrier{out_memory_config, topology, input_tensor.get_workers()});
}

}  // namespace ttnn::operations::ccl
