// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hang_device_operation.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::prim {
ExecuteTestHangDeviceOperation::SingleCore::cached_program_t ExecuteTestHangDeviceOperation::SingleCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args;
    auto& output_tensor = tensor_return_value;
    constexpr CoreCoord core = {0, 0};

    tt::tt_metal::Program program{};

    tt::tt_metal::IDevice* device = input_tensor.tensor.device();

    std::vector<uint32_t> compute_kernel_args = {};
    KernelHandle void_compute_kernel_id = CreateKernel(
        program,
#if TTNN_OPERATION_TIMEOUT_SECONDS > 0
        "ttnn/cpp/ttnn/operations/experimental/test/hang_device/device/kernels/compute/hang_device_kernel.cpp",
#else
        "ttnn/cpp/ttnn/operations/experimental/test/hang_device/device/kernels/compute/non_hang_device_kernel.cpp",
#endif
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
            .opt_level = KernelBuildOptLevel::O3});

    return {std::move(program), {}};
}

void ExecuteTestHangDeviceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {}

}  // namespace ttnn::prim
