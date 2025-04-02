// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sort_program_factory.hpp"

namespace ttnn::operations::experimental::reduction::sort::program {

SortProgramFactory::cached_program_t SortProgramFactory::create(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors) {
    tt::tt_metal::Program program{};

    // TODO: Implementation in next PR
    tt::log_warning("sort_program_interleaved not implemented yet!");

    return {std::move(program), {}};
}

void SortProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensors) {
    // TODO: Implement the logic to override runtime arguments for the cached program
}

}  // namespace ttnn::operations::experimental::reduction::sort::program
