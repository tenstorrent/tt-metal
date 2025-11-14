// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gate.hpp"
#include "device/gate_op.hpp"

namespace ttnn::operations::experimental::deepseek_b1::gate {

ttnn::Tensor GateOperation::invoke(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const ttnn::Tensor& expert_bias,
    const ttnn::CoreGrid& core_grid,
    const std::optional<const ttnn::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DataType>& dtype,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
    GateDeviceOperation device_op{
        .program_config =
            GateProgramConfig{
                .compute_with_storage_grid_size = CoreCoord(core_grid.x, core_grid.y),
            },
        .output_mem_config = memory_config,
        .output_dtype = dtype,
        .compute_kernel_config = compute_kernel_config,
    };

    return tt::tt_metal::operation::run(device_op, {a, b, expert_bias}).at(0);
}

}  // namespace ttnn::operations::experimental::deepseek_b1::gate
