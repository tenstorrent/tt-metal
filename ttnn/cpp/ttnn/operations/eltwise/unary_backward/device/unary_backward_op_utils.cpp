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
    }
    TT_THROW("Unary backward op type {} has no kernel spec", static_cast<int>(op_type));
}

std::string_view to_string(UnaryBackwardOpType op_type) {
    switch (op_type) {
        case UnaryBackwardOpType::SIGMOID_BW: return "SIGMOID_BW";
    }
    TT_THROW("Unary backward op type {} has no name", static_cast<int>(op_type));
}

}  // namespace ttnn::operations::unary_backward
