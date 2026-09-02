// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ApplyTwiddlesXlFactory — ProgramSpec program factory for the
// apply_twiddles_xl op (large-modulus between-pass elementwise complex
// multiply for the three-pass composite).  Reuses apply_twiddles_compute
// and apply_twiddles_writer verbatim; only the reader kernel changes
// (apply_twiddles_xl_reader.cpp) to build the twiddle row on-the-fly
// from a small per-(device, big_modulus, full_N) delta lookup table.

#pragma once

#include "ttnn/metal_v2_artifacts.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "apply_twiddles_xl_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct ApplyTwiddlesXlFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ApplyTwiddlesXlParams& operation_attributes,
        const ApplyTwiddlesXlTensorArgs& tensor_args,
        std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
