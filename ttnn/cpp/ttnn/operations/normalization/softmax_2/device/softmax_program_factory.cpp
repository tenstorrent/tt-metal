// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <utility>

namespace ttnn::operations::normalization::softmax::program {
// General-purpose softmax with arbitrary dimension support
SoftmaxProgramFactoryGeneral::cached_program_t SoftmaxProgramFactoryGeneral::create(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    tt::tt_metal::Program program{};

    return {std::move(program), {}};
}

void SoftmaxProgramFactoryGeneral::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    // Implementation details...
}

// Optimized for transformer attention patterns
SoftmaxProgramFactoryAttentionOptimized::cached_program_t SoftmaxProgramFactoryAttentionOptimized::create(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    tt::tt_metal::Program program{};

    return {std::move(program), {}};
}

void SoftmaxProgramFactoryAttentionOptimized::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    // Implementation details...
}
}  // namespace ttnn::operations::normalization::softmax::program
