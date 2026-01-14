// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"

#include "sdpa_decode_device_operation_types.hpp"

namespace ttnn::operations::transformer::sdpa_decode::program {

struct SdpaDecodeProgramFactory {
    struct shared_variables_t {
        uint32_t num_active_cores = 0;
        std::vector<CoreCoord> core_group;
        tt::tt_metal::KernelHandle reader_kernels_id{};
        tt::tt_metal::KernelHandle writer_kernels_id{};
        tt::tt_metal::KernelHandle compute_kernels_id{};
        uint32_t num_cores_per_batch = 0;
        uint32_t num_cores_per_head = 0;
        uint32_t num_output_cores = 0;
        tt::tt_metal::CBHandle cb_in8_id{};
        tt::tt_metal::CBHandle cb_in9_id{};
        bool is_output_sharded = false;
        tt::tt_metal::CBHandle cb_out4_id{};
        uint32_t B = 0;
        uint32_t q_heads_parallel_factor = 0;
        bool use_cur_pos_tensor = false;
        bool use_attention_mask = false;
        bool use_attention_sink = false;
        bool is_paged_attention = false;
        bool is_causal = false;
        bool use_mla = false;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::transformer::sdpa_decode::program
