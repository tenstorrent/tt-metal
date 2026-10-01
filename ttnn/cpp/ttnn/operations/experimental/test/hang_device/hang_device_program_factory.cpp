// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hang_device_operation.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::prim {

// The kernel takes no runtime args, no DFBs and no tensors, so there is nothing to bind and a cache
// hit has nothing to refresh. That is the intended shape here: the spec is a single kernel with no
// per-dispatch state, and the op exists to hang the device.
ttnn::device_operation::ProgramArtifacts
ExecuteTestHangDeviceOperation::ExecuteTestHangDeviceOperationProgramFactory::create_program_artifacts(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    constexpr CoreCoord core = {0, 0};

    const KernelSpecName COMPUTE{"compute"};

    ProgramSpec spec;
    spec.name = "hang_device";

    spec.kernels.push_back(KernelSpec{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/test/hang_device/device/kernels/compute/hang_device_kernel.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .hw_config =
            ComputeHardwareConfig{
                .fpu_math_fidelity = MathFidelity::HiFi4,
                .sfpu_precision_mode = Precision::Precise,  // legacy math_approx_mode = false
                .enable_32_bit_dest = false,
            },
    });

    spec.work_units.push_back(WorkUnitSpec{
        .name = "hang_device",
        .kernels = {COMPUTE},
        .target_nodes = CoreRangeSet(CoreRange(core, core)),
    });

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = ProgramRunArgs{},
    };
}

}  // namespace ttnn::prim
