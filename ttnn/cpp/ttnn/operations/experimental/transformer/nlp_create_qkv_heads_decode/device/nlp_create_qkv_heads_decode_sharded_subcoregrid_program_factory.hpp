// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "nlp_create_qkv_heads_decode_device_operation_types.hpp"

namespace ttnn::operations::experimental::nlp_create_qkv_heads_decode::program {

struct NLPCreateQKVHeadsDecodeShardedSubcoregridProgramFactory {
    using operation_attributes_t = NlpCreateQkvHeadsDecodeParams;
    using tensor_args_t = NlpCreateQkvHeadsDecodeInputs;
    using tensor_return_value_t = nlp_create_qkv_heads_decode::tensor_return_value_t;

    struct shared_variables_t {
        tt::tt_metal::KernelHandle q_reader_kernel_id{};
        tt::tt_metal::KernelHandle q_writer_kernel_id{};
        tt::tt_metal::KernelHandle k_reader_kernel_id{};
        tt::tt_metal::KernelHandle k_writer_kernel_id{};
        uint32_t q_num_cores{};
        uint32_t k_num_cores{};
        tt::tt_metal::CBHandle cb_q_output{};
        tt::tt_metal::CBHandle cb_k_output{};
        tt::tt_metal::CBHandle cb_v_output{};
        std::vector<CoreCoord> q_cores_vector;
        std::vector<CoreCoord> k_cores_vector;
        uint32_t element_size{};
        uint32_t sub_tile_line_bytes{};
        bool overlap_qk_coregrid{};
        bool use_batch_offset{};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output_tensors);
};

}  // namespace ttnn::operations::experimental::nlp_create_qkv_heads_decode::program
