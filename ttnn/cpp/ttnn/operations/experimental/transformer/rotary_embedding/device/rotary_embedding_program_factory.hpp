// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingSharedVariables {
    tt::tt_metal::KernelHandle unary_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle unary_writer_kernel_id = 0;
    tt::tt_metal::CBHandle cb_input{};
    tt::tt_metal::CBHandle cb_output{};
    std::vector<CoreCoord> cores;
    uint32_t g1_numcores = 0;
    uint32_t num_rows_per_core_group_1 = 0;
    uint32_t num_rows_per_core_group_2 = 0;
    uint32_t Wbytes = 0;
    uint32_t Wt = 0;
    uint32_t HtWt = 0;
};

struct RotaryEmbeddingProgramFactory {
    using shared_variables_t = RotaryEmbeddingSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const RotaryEmbeddingParams& operation_attributes,
        const RotaryEmbeddingInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const RotaryEmbeddingParams& operation_attributes,
        const RotaryEmbeddingInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
