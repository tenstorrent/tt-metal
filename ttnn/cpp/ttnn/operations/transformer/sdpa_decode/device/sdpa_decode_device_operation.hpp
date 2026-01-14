// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "sdpa_decode_device_operation_types.hpp"
#include "sdpa_decode_program_factory.hpp"

namespace ttnn::operations::transformer::sdpa_decode {

struct SdpaDecodeDeviceOperation {
    using operation_attributes_t = sdpa_decode::operation_attributes_t;
    using tensor_args_t = sdpa_decode::tensor_args_t;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    using program_factory_t = std::variant<program::SdpaDecodeProgramFactory>;
    using shared_variables_t = program::SdpaDecodeProgramFactory::shared_variables_t;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::operations::transformer::sdpa_decode

namespace ttnn::prim {

ttnn::operations::transformer::sdpa_decode::SdpaDecodeDeviceOperation::tensor_return_value_t sdpa_decode(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const std::optional<const Tensor>& input_tensor_v,
    const std::optional<const Tensor>& cur_pos_tensor,
    const std::optional<const Tensor>& page_table_tensor,
    const std::optional<const Tensor>& attn_mask,
    const std::optional<const Tensor>& attention_sink,
    bool is_causal,
    bool paged_attention,
    const std::vector<uint32_t>& cur_pos,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    uint32_t k_chunk_size,
    std::optional<bool> share_cache,
    std::optional<bool> use_mla,
    std::optional<uint32_t> head_dim_v);

}  // namespace ttnn::prim
