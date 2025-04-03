// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>

#include "device/fast_reduce_nc_device_operation.hpp"
#include <tt_stl/span.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

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

constexpr auto fast_reduce_nc = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::fast_reduce_nc",
    ttnn::operations::experimental::reduction::FastReduceNCOperation>();

}  // namespace experimental::reduction

}  // namespace ttnn
