// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "deltanet_full_program_factory.hpp"
#include "deltanet_full_device_operation_types.hpp"

namespace ttnn::operations::experimental::deltanet {

struct DeltaNetDecodeFullDeviceOperation {
    using operation_attributes_t = DeltaNetDecodeFullParams;
    using tensor_args_t = DeltaNetDecodeFullInputs;

    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    using program_factory_t = std::variant<DeltaNetDecodeFullProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args);
};

}  // namespace ttnn::operations::experimental::deltanet

namespace ttnn::prim {

std::vector<Tensor> deltanet_decode_full(
    const Tensor& qkv_proj,
    const Tensor& z_proj,
    const Tensor& b_proj,
    const Tensor& a_proj,
    const Tensor& conv_state,
    const Tensor& recurrent_state,
    const Tensor& conv1d_weight,
    const Tensor& a_log,
    const Tensor& dt_bias,
    const Tensor& norm_weight,
    uint32_t num_heads,
    uint32_t num_k_heads,
    uint32_t k_head_dim,
    uint32_t v_head_dim,
    uint32_t conv_dim,
    uint32_t conv_kernel_size,
    uint32_t head_expand_ratio,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);

}  // namespace ttnn::prim
