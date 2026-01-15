// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include "ttnn/device_operation.hpp"
#include "rotary_embedding_llama_device_operation_types.hpp"

namespace ttnn::operations::experimental::transformer::rotary_embedding_llama::program {

struct RotaryEmbeddingLlamaMultiCore {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle unary_reader_kernel_id{};
        tt::tt_metal::KernelHandle unary_writer_kernel_id{};
        tt::tt_metal::KernelHandle rotary_embedding_kernel_id{};
        std::vector<CoreCoord> cores;
        uint32_t num_active_cores{};

        tt::tt_metal::CBHandle cb_input{};
        tt::tt_metal::CBHandle cb_cos{};
        tt::tt_metal::CBHandle cb_sin{};
        tt::tt_metal::CBHandle cb_trans_mat{};
        tt::tt_metal::CBHandle cb_output{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const RotaryEmbeddingLlamaParams& operation_attributes,
        const RotaryEmbeddingLlamaInputs& tensor_args,
        tt::tt_metal::Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const RotaryEmbeddingLlamaParams& operation_attributes,
        const RotaryEmbeddingLlamaInputs& tensor_args,
        tt::tt_metal::Tensor& output);
};

}  // namespace ttnn::operations::experimental::transformer::rotary_embedding_llama::program
