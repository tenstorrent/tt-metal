// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include "ttnn/device_operation.hpp"
#include "rotary_embedding_llama_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// Program factory for the decode-mode HC-transpose case.
// Input shape:  [1, num_heads, batch_size, head_dim]  (interleaved)
// Output shape: [1, num_heads, batch_size, head_dim]  (interleaved)
// Cos/sin:      [1, num_heads_cs, batch_size, head_dim] (num_heads_cs == 1 or num_heads)
struct RotaryEmbeddingLlamaHCDecode {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        tt::tt_metal::KernelHandle compute_kernel_id{};
        std::vector<CoreCoord> cores;
        uint32_t num_active_cores{};
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

}  // namespace ttnn::experimental::prim
