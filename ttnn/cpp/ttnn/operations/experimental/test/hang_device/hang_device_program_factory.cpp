// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hang_device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

// The kernel takes no runtime args and no CBs, so there is nothing to bind and cache hits re-run
// this function. That is the intended shape here: the descriptor is a single kernel with no
// per-dispatch state, and the op exists to hang the device.
tt::tt_metal::ProgramDescriptor ExecuteTestHangDeviceOperation::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    using namespace tt::tt_metal;

    constexpr CoreCoord core = {0, 0};

    ProgramDescriptor desc;

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/test/hang_device/device/kernels/compute/hang_device_kernel.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = CoreRangeSet(CoreRange(core, core));
    compute_desc.opt_level = KernelBuildOptLevel::O3;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .math_approx_mode = false,
    };

    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
