// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn::operations::experimental::speculative_execution::detail {

operation::ProgramWithCallbacks swap_tensor(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    uint32_t num_links,
    uint32_t num_devices,
    uint32_t device_index,
    ttnn::ccl::Topology topology,
    GlobalSemaphore global_semaphore,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device);

}  // namespace ttnn::operations::experimental::speculative_execution::detail
