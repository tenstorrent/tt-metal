// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::bringup::rms_norm_ttnn {

// The op's attributes: everything the program depends on besides the tensors' specs.
// The compute config is the RESOLVED ComputeConfigDescriptor (rms_norm_ttnn.py's
// normalize_compute_kernel_config), which the compute kernel receives unmodified.
struct RmsNormParams {
    double epsilon = 1e-12;
    tt::tt_metal::ComputeConfigDescriptor compute_config;
    // resolve_program_config(): the pass-B DEST sub-block (0 == the op's own choice) and `inplace`.
    uint32_t subblock_w = 0;
    bool inplace = false;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct RmsNormInputs {
    Tensor input;
    std::optional<Tensor> weight;
    std::optional<Tensor> bias;
    std::optional<Tensor> residual;
};

// The refusals keep the Python op's exception TYPES.  std::invalid_argument surfaces in Python as
// ValueError (nanobind's built-in translation); UnsupportedAxisError is translated by the binding to
// ttnn.operations._op_contract.UnsupportedAxisValue (a NotImplementedError); NotImplementedErrorCpp
// to a plain NotImplementedError.
struct ValueErrorCpp : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};
struct UnsupportedAxisError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct NotImplementedErrorCpp : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace ttnn::operations::bringup::rms_norm_ttnn
