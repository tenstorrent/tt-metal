// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "embedding_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct EmbeddingsRMProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id {};
        tt::tt_metal::KernelHandle writer_kernel_id {};
        std::vector<tt::tt_metal::CoreCoord> cores;
        tt::tt_metal::CBHandle cb_out {};
        bool output_sharded = false;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const EmbeddingParams& operation_attributes, const EmbeddingInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const EmbeddingParams& operation_attributes,
        const EmbeddingInputs& tensor_args,
        Tensor& tensor_return_value);
};
}  // namespace ttnn::prim
