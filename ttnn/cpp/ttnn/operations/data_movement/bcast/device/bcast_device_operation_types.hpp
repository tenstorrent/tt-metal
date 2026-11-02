// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"

namespace ttnn::prim {

struct BcastParams {
    ttnn::BcastOpMath math_op;
    ttnn::BcastOpDim dim;
    tt::tt_metal::MemoryConfig output_mem_config;
    bool in_place = false;
};

struct BcastInputs {
    Tensor input_a;
    Tensor input_b;
    std::optional<Tensor> preallocated_output;
};

// Compute config for the bcast kernels. float32 operands need a 32-bit Dest, otherwise the sum or
// product is rounded to 16 bits before it is packed. Metal 2.0 requires an explicit unpack mode for
// each consumed float32 DFB when the 32-bit Dest is on; UnpackToSrc is what the FPU path needs.
inline tt::tt_metal::experimental::ComputeHardwareConfig make_bcast_compute_hw_config(
    const Tensor& a,
    const Tensor& b,
    const Tensor& output,
    const tt::tt_metal::experimental::DFBSpecName& in0,
    const tt::tt_metal::experimental::DFBSpecName& in1) {
    tt::tt_metal::experimental::ComputeHardwareConfig config;  // legacy ComputeConfigDescriptor{} defaults
    const bool a_fp32 = a.dtype() == DataType::FLOAT32;
    const bool b_fp32 = b.dtype() == DataType::FLOAT32;
    if (a_fp32 || b_fp32 || output.dtype() == DataType::FLOAT32) {
        config.enable_32_bit_dest = true;
        if (a_fp32) {
            config.unpack_modes.emplace(in0, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
        if (b_fp32) {
            config.unpack_modes.emplace(in1, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
    }
    return config;
}

}  // namespace ttnn::prim
