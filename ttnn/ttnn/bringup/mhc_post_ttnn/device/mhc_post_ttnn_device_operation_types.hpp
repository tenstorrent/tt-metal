// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdexcept>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::mhc_post_ttnn {

// The op's attributes: everything the program depends on besides the tensors' specs. The compute config is the
// resolved one (the caller's, or default_compute_kernel_config()); the builder copies math_fidelity,
// fp32_dest_acc_en and math_approx_mode from it, as mhc_post_program_descriptor.py does.
struct MhcPostParams {
    tt::tt_metal::ComputeConfigDescriptor compute_config;
};

struct MhcPostInputs {
    Tensor input;     // F   (..., T, C)      the sublayer output
    Tensor residual;  // X   (..., T, n*C)    the n residual streams
    Tensor post;      // (..., T, n)          float32
    Tensor comb;      // (..., T, n*n)        float32, comb[i*n + j]
};

// The refusals keep the Python op's exception types: std::invalid_argument surfaces as ValueError (nanobind's
// built-in translation); UnsupportedAxisError is translated by the binding to
// ttnn.operations._op_contract.UnsupportedAxisValue.
struct ValueErrorCpp : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};
struct UnsupportedAxisError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace ttnn::operations::bringup::mhc_post_ttnn
