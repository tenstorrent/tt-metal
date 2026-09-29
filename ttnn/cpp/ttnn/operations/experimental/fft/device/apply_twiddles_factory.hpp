// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ApplyTwiddlesFactory — ProgramSpec program factory for the
// apply_twiddles op (between-pass elementwise complex multiply step of
// Cooley–Tukey two-pass FFT).  Multi-core, supports fp32 and bf16, uses
// per-(N1, N2) cached twiddle tensors bound as named TensorParameters.

#pragma once

#include "ttnn/metal_v2_artifacts.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "apply_twiddles_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct ApplyTwiddlesFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ApplyTwiddlesParams& attrs,
        const ApplyTwiddlesTensorArgs& tensor_args,
        std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
