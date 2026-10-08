// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_backward_op_utils.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::operations::unary_backward {

using tt::tt_metal::MathFidelity;

const UnaryBackwardKernelSpec& get_kernel_spec(UnaryBackwardOpType op_type) {
    switch (op_type) {
        case UnaryBackwardOpType::SIGMOID_BW: {
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_sigmoid.cpp",
                .math_fidelity = MathFidelity::HiFi4,
                // d/dx sigmoid = s(1 - s). For x beyond about +/-6 a bfloat16 s rounds to
                // within one ulp of 1.0, and (1 - s) then keeps at most a couple of significant
                // bits -- the error the composite has always had here. Holding s at float32
                // through the subtraction removes it.
                .force_fp32_dest_acc = true,
            };
            return spec;
        }
        case UnaryBackwardOpType::TANH_BW: {
            // d/dx tanh = sech^2(x), computed directly by TanhDerivative rather than as
            // 1 - tanh^2, so there is no cancellation to hold at float32 and DEST follows the
            // operand dtypes.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_tanh.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::GELU_BW: {
            // GeluVariant::ACCURATE: the exact-GELU derivative by polynomial.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_gelu_poly.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::GELU_TANH_BW: {
            // GeluVariant::TANH. The chain keeps six tiles live in DEST, so a float32 DEST (four
            // slots) needs the reordered variant.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_gelu_tanh.cpp",
                .compute_kernel_path_fp32_dest =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_gelu_tanh_fp32.cpp",
            };
            return spec;
        }
    }
    TT_THROW("Unary backward op type {} has no kernel spec", static_cast<int>(op_type));
}

std::string_view to_string(UnaryBackwardOpType op_type) {
    switch (op_type) {
        case UnaryBackwardOpType::SIGMOID_BW: return "SIGMOID_BW";
        case UnaryBackwardOpType::TANH_BW: return "TANH_BW";
        case UnaryBackwardOpType::GELU_BW: return "GELU_BW";
        case UnaryBackwardOpType::GELU_TANH_BW: return "GELU_TANH_BW";
    }
    TT_THROW("Unary backward op type {} has no name", static_cast<int>(op_type));
}

}  // namespace ttnn::operations::unary_backward
