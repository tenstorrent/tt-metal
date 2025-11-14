// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tt_metal.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/global_circular_buffer.hpp>

namespace ttnn::operations::experimental::deepseek_b1::gate {

struct gate_common_override_variables_t {
    std::vector<tt::tt_metal::KernelHandle> kernels;
    std::vector<tt::tt_metal::CBHandle> cbs;
    bool extract_shard_sub_blocks;
    CoreCoord start_core;
    std::vector<CoreCoord> cores;
    uint32_t num_cores_with_work;
};

// Gate program factory
tt::tt_metal::operation::ProgramWithCallbacks deepseek_b1_gate(
    const Tensor& a,
    const Tensor& b,
    const Tensor& expert_bias,
    const Tensor& output_tensor,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config);

}  // namespace ttnn::operations::experimental::deepseek_b1::gate
