// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "embedding_backward_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

struct EmbeddingBackwardProgramFactory {
    struct shared_variables_t {
        KernelHandle reader_kernel_id{};
        std::vector<CoreCoord> cores;
        distributed::MeshDevice* device = nullptr;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const EmbeddingBackwardParams& operation_attributes,
        const EmbeddingBackwardInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const EmbeddingBackwardParams& operation_attributes,
        const EmbeddingBackwardInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
