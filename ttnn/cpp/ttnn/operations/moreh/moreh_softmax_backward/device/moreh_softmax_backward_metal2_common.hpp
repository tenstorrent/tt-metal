// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <initializer_list>
#include <utility>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "ttnn/operations/moreh/moreh_softmax_backward/device/moreh_softmax_backward_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_softmax_backward {

// Metal 2.0 named resources and small builders shared by all five moreh_softmax_backward
// factories.
//
// These live in one header with inline linkage rather than in per-factory anonymous namespaces
// because ttnn_op_moreh is a unity build: the anonymous namespaces of the five factory .cpp files
// merge into one scope, where same-named constants would collide.
namespace metal2 {

using tt::tt_metal::experimental::ComputeUnpackModes;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::TensorParamName;

// ---------------------------------------------------------------------------------------------
// Dataflow buffers, named for the value each one carries
// ---------------------------------------------------------------------------------------------
inline const DFBSpecName Y_DFB{"y"};                // softmax output
inline const DFBSpecName DY_DFB{"dy"};              // incoming gradient
inline const DFBSpecName SCALER_DFB{"scaler"};      // reduction scaler tile
inline const DFBSpecName MASK_DFB{"mask"};          // partial-tile mask
inline const DFBSpecName DX_DFB{"dx"};              // outgoing gradient
inline const DFBSpecName YDY_DFB{"ydy"};            // y * dy
inline const DFBSpecName SUM_DFB{"sum"};            // sum(y * dy)
inline const DFBSpecName DY_M_SUM_DFB{"dy_m_sum"};  // dy - sum
inline const DFBSpecName ADD_DFB{"add"};            // running add; the large W/H variants only

// ---------------------------------------------------------------------------------------------
// Tensor parameters, named for the op's own tensor_args_t fields
// ---------------------------------------------------------------------------------------------
inline const TensorParamName OUTPUT_TENSOR{"output"};
inline const TensorParamName OUTPUT_GRAD_TENSOR{"output_grad"};
inline const TensorParamName INPUT_GRAD_TENSOR{"input_grad"};

// ---------------------------------------------------------------------------------------------
// Kernels
//
// The compute kernel is specialized per core group: one KernelSpec per group, each baking in
// that group's per-core tile count as a compile-time argument, placed by its own WorkUnitSpec
// over that group's (disjoint) nodes.
// ---------------------------------------------------------------------------------------------
inline const KernelSpecName READER_KERNEL{"reader"};
inline const KernelSpecName WRITER_KERNEL{"writer"};
inline const KernelSpecName COMPUTE_KERNEL_G1{"compute_g1"};
inline const KernelSpecName COMPUTE_KERNEL_G2{"compute_g2"};

// ---------------------------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------------------------

// tile_format_metadata is deliberately left unset: no buffer in this op uses non-default tile
// geometry, and for standard 32x32 tiles the JIT fallback produces the same descriptors as
// setting it would.
inline DataflowBufferSpec MakeDFB(
    const DFBSpecName& unique_id, uint32_t num_entries, uint32_t entry_size, tt::DataFormat data_format) {
    return DataflowBufferSpec{
        .unique_id = unique_id,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = data_format,
    };
}

// SOFTMAX and LOG select paths inside the compute kernels. SOFTMIN and FP32_DEST_ACC_EN are
// preserved for descriptor parity even though these kernels do not read them. The readers of the
// large W/H/C variants consume LOG as well and build their defines separately.
inline KernelSpec::CompilerOptions::Defines MakeComputeDefines(MorehSoftmaxBackwardOp op, bool fp32_dest_acc_en) {
    KernelSpec::CompilerOptions::Defines defines;
    if (op == MorehSoftmaxBackwardOp::SOFTMAX) {
        defines.emplace("SOFTMAX", "1");
    } else {
        defines.emplace("SOFTMIN", "1");
    }
    if (op == MorehSoftmaxBackwardOp::LOGSOFTMAX) {
        defines.emplace("LOG", "1");
    }
    if (fp32_dest_acc_en) {
        defines.emplace("FP32_DEST_ACC_EN", "1");
    }
    return defines;
}

// Unpack modes for the compute kernel, keyed by the DFBs it consumes.
//
// This op asks for the default unpack path on every buffer (UnpackToSrc). That is normally said
// by omitting the entry — but an explicit choice is *required* for a Float32 buffer consumed by a
// kernel with a 32-bit Dest register, which is precisely the case fp32_dest_acc_en creates here.
// So spell out the default for those, and only those.
//
// Pass every DFB the compute kernel consumes, with the format its spec declares. A self-looped
// buffer counts as consumed (its CONSUMER binding is what the requirement keys on); a buffer the
// kernel only produces does not.
inline ComputeUnpackModes MakeUnpackModes(
    bool fp32_dest_acc_en, std::initializer_list<std::pair<DFBSpecName, tt::DataFormat>> consumed_dfbs) {
    ComputeUnpackModes unpack_modes;
    if (!fp32_dest_acc_en) {
        return unpack_modes;
    }
    for (const auto& [dfb, data_format] : consumed_dfbs) {
        if (data_format == tt::DataFormat::Float32) {
            unpack_modes.emplace(dfb, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
    }
    return unpack_modes;
}

}  // namespace metal2

}  // namespace ttnn::operations::moreh::moreh_softmax_backward
