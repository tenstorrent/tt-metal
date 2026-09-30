// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <stdexcept>
#include <tuple>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::mhc_pre_ttnn {

// The op's attributes: everything the program depends on besides the tensors' specs. n is derived from the weight
// (n (n + 2) = its last dim); the compute config is the resolved one (the caller's or default_compute_kernel_config()).
struct MhcPreParams {
    uint32_t n = 4;
    // Python floats (double); each is rounded to fp32 once, when packed into the runtime args, as the Python op does.
    std::array<double, 3> scale{};  // (a_pre, a_post, a_res)
    uint32_t sinkhorn_iters = 20;
    double eps = 1e-6;
    double norm_eps = 1e-6;
    tt::tt_metal::ComputeConfigDescriptor compute_config;
};

struct MhcPreInputs {
    Tensor input;        // X  (..., T, n*C)
    Tensor proj_weight;  // W  (n*C, n (n + 2))
    Tensor proj_bias;    // b  (1, n (n + 2)) float32
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

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn
