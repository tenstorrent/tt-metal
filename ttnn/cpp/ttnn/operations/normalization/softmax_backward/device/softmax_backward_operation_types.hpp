// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/kernel_types.hpp>

#include <cstdint>

namespace ttnn::operations::normalization::softmax_backward {

struct operation_attributes_t {
    const uint32_t dim;
};

struct tensor_args_t {
    ttnn::Tensor softmax_output;
    ttnn::Tensor upstream_grad;
};

using spec_return_value_t = ttnn::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

// Shared variables used by both non-streaming and streaming factories
// Only stores kernel handles needed for override_runtime_arguments
struct shared_variables_t {
    tt::tt_metal::KernelHandle unary_reader_kernel_id;
    tt::tt_metal::KernelHandle unary_writer_kernel_id;
};

}  // namespace ttnn::operations::normalization::softmax_backward
