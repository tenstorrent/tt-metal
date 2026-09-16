// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_backward_op_utils.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::operations::binary_backward {

namespace {

constexpr BinaryBackwardKernelSpec kMulBw{
    .compute_kernel_path = "ttnn/cpp/ttnn/operations/eltwise/binary_backward/device/kernels/compute/eltwise_bw_mul.cpp",
    .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
    .force_fp32_dest_acc = false,
    .num_outputs = 2,
};

}  // namespace

const BinaryBackwardKernelSpec& kernel_spec(BinaryBackwardOpType op_type) {
    switch (op_type) {
        case BinaryBackwardOpType::MUL_BW: return kMulBw;
    }
    TT_THROW("binary_backward: unknown BinaryBackwardOpType={}", static_cast<uint32_t>(op_type));
}

}  // namespace ttnn::operations::binary_backward
