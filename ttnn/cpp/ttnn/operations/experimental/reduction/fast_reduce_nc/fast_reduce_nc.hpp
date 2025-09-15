// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/fast_reduce_nc_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {
namespace operations::experimental::reduction {

struct FastReduceNCOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input,
        tt::stl::Span<const int32_t> dims,
        const std::optional<const Tensor>& output,
        const ttnn::MemoryConfig& memory_config,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config);
};

}  // namespace operations::experimental::reduction

namespace experimental::reduction {

constexpr auto fast_reduce_nc = ttnn::register_operation<
    "ttnn::experimental::fast_reduce_nc",
    ttnn::operations::experimental::reduction::FastReduceNCOperation>();

}  // namespace experimental::reduction

}  // namespace ttnn
