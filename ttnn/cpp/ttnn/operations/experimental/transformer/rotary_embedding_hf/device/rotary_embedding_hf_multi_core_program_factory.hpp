// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/rotary_embedding_hf/device/rotary_embedding_hf_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

struct RotaryEmbeddingHfMultiCore {
    struct shared_variables_t {
        KernelHandle unary_reader_kernel_id;
        KernelHandle unary_writer_kernel_id;
        CBHandle cb_input;
        CBHandle cb_output;
        std::vector<CoreCoord> cores;
        uint32_t g1_numcores;
        uint32_t num_rows_per_core_group_1;
        uint32_t num_rows_per_core_group_2;
        uint32_t Wt;
        uint32_t Ht;
        uint32_t HtWt;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const RotaryEmbeddingHfParams& operation_attributes,
        const RotaryEmbeddingHfInputs& tensor_args,
        Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const RotaryEmbeddingHfParams& operation_attributes,
        const RotaryEmbeddingHfInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::experimental::prim
