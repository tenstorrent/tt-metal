// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "embedding_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::embedding::program {

struct EmbeddingsTilizedIndicesProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id {};
        tt::tt_metal::KernelHandle writer_kernel_id {};
        std::vector<tt::tt_metal::CoreCoord> cores;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const embedding::EmbeddingParams& operation_attributes,
        const embedding::EmbeddingInputs& tensor_args,
        embedding::tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const embedding::EmbeddingParams& operation_attributes,
        const embedding::EmbeddingInputs& tensor_args,
        embedding::tensor_return_value_t& tensor_return_value);
};
}  // namespace ttnn::operations::embedding::program
