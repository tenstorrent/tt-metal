// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"

#include "sdpa_decode_device_operation_types.hpp"

// namespace ttnn::operations::transformer::detail {

// tt::tt_metal::operation::ProgramWithCallbacks sdpa_decode_multi_core(
//     const Tensor& input_tensor_q,
//     const Tensor& input_tensor_k,
//     const Tensor& input_tensor_v,
//     std::optional<const Tensor> cur_pos_tensor,
//     std::optional<const Tensor> page_table_tensor,
//     std::optional<const Tensor> attn_mask,
//     std::optional<const Tensor> attention_sink,
//     const Tensor& output_tensor,
//     bool is_causal,
//     const std::vector<uint32_t>& cur_pos_ids,
//     std::optional<float> scale,
//     DeviceComputeKernelConfig compute_kernel_config,
//     std::optional<SDPAProgramConfig> program_config,
//     uint32_t k_chunk_size,
//     std::optional<bool> share_cache,
//     bool mla = false,
//     uint32_t head_dim_v = 0,
//     std::optional<uint32_t> sliding_window_size = std::nullopt);

// }  // namespace ttnn::operations::transformer::detail

namespace ttnn::operations::transformer::sdpa_decode::program {

struct SdpaDecodeProgramFactory {
    struct shared_variables_t {
        uint32_t num_active_cores;
        std::vector<CoreCoord> core_group;
        tt::tt_metal::KernelHandle reader_kernels_id;
        tt::tt_metal::KernelHandle writer_kernels_id;
        tt::tt_metal::KernelHandle compute_kernels_id;
        uint32_t num_cores_per_batch;
        uint32_t num_cores_per_head;
        uint32_t num_output_cores;
        tt::tt_metal::CBHandle cb_in8_id;
        tt::tt_metal::CBHandle cb_in9_id;
        bool is_output_sharded;
        tt::tt_metal::CBHandle cb_out4_id;
        uint32_t B;
        uint32_t q_heads_parallel_factor;
        bool use_cur_pos_tensor;
        bool use_attention_mask;
        bool use_attention_sink;
        bool is_paged_attention;
        bool is_causal;
        bool use_mla;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::transformer::sdpa_decode::program
