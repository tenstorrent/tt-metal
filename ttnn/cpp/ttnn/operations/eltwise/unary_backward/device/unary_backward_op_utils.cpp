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
        case UnaryBackwardOpType::CELU_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_celu.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::SELU_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_selu.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::LOG_SIGMOID_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_log_sigmoid.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::SOFTPLUS_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_softplus.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::HARDSIGMOID_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_hardsigmoid.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::HARDTANH_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_hardtanh.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::LEAKY_RELU_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_leaky_relu.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::RELU6_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_relu6.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::HARDSHRINK_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_hardshrink.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::SOFTSHRINK_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_softshrink.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::ABS_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_abs.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::ACOS_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_acos.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::ASIN_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_asin.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::ATANH_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_atanh.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::ASINH_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_asinh.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::LOGIT_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_logit.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::LOGITEPS_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_logiteps.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::SQRT_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_sqrt.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::RSQRT_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_rsqrt.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::LOG_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_log.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::LOG2_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_log2.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::LOG10_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_log10.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::LOG1P_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_log1p.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::RECIPROCAL_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_reciprocal.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::EXPM1_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_expm1.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::EXP2_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_exp2.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::SQUARE_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_square.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::SINH_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_sinh.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::COSH_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_cosh.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::ERFINV_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_erfinv.cpp",
            };
            return spec;
        }
        case UnaryBackwardOpType::MULTIGAMMALN_BW: {
            // The generated kernel evaluates the gradient over BF16 DEST.
            static const UnaryBackwardKernelSpec spec{
                .compute_kernel_path =
                    "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/compute/"
                    "eltwise_bw_multigammaln.cpp",
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
        case UnaryBackwardOpType::CELU_BW: return "CELU_BW";
        case UnaryBackwardOpType::SELU_BW: return "SELU_BW";
        case UnaryBackwardOpType::LOG_SIGMOID_BW: return "LOG_SIGMOID_BW";
        case UnaryBackwardOpType::SOFTPLUS_BW: return "SOFTPLUS_BW";
        case UnaryBackwardOpType::HARDSIGMOID_BW: return "HARDSIGMOID_BW";
        case UnaryBackwardOpType::HARDTANH_BW: return "HARDTANH_BW";
        case UnaryBackwardOpType::LEAKY_RELU_BW: return "LEAKY_RELU_BW";
        case UnaryBackwardOpType::RELU6_BW: return "RELU6_BW";
        case UnaryBackwardOpType::HARDSHRINK_BW: return "HARDSHRINK_BW";
        case UnaryBackwardOpType::SOFTSHRINK_BW: return "SOFTSHRINK_BW";
        case UnaryBackwardOpType::ABS_BW: return "ABS_BW";
        case UnaryBackwardOpType::ACOS_BW: return "ACOS_BW";
        case UnaryBackwardOpType::ASIN_BW: return "ASIN_BW";
        case UnaryBackwardOpType::ATANH_BW: return "ATANH_BW";
        case UnaryBackwardOpType::ASINH_BW: return "ASINH_BW";
        case UnaryBackwardOpType::LOGIT_BW: return "LOGIT_BW";
        case UnaryBackwardOpType::LOGITEPS_BW: return "LOGITEPS_BW";
        case UnaryBackwardOpType::SQRT_BW: return "SQRT_BW";
        case UnaryBackwardOpType::RSQRT_BW: return "RSQRT_BW";
        case UnaryBackwardOpType::LOG_BW: return "LOG_BW";
        case UnaryBackwardOpType::LOG2_BW: return "LOG2_BW";
        case UnaryBackwardOpType::LOG10_BW: return "LOG10_BW";
        case UnaryBackwardOpType::LOG1P_BW: return "LOG1P_BW";
        case UnaryBackwardOpType::RECIPROCAL_BW: return "RECIPROCAL_BW";
        case UnaryBackwardOpType::EXPM1_BW: return "EXPM1_BW";
        case UnaryBackwardOpType::EXP2_BW: return "EXP2_BW";
        case UnaryBackwardOpType::SQUARE_BW: return "SQUARE_BW";
        case UnaryBackwardOpType::SINH_BW: return "SINH_BW";
        case UnaryBackwardOpType::COSH_BW: return "COSH_BW";
        case UnaryBackwardOpType::ERFINV_BW: return "ERFINV_BW";
        case UnaryBackwardOpType::MULTIGAMMALN_BW: return "MULTIGAMMALN_BW";
    }
    TT_THROW("Unary backward op type {} has no name", static_cast<int>(op_type));
}

}  // namespace ttnn::operations::unary_backward
