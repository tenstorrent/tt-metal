// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/device_operation.hpp"
#include "create_qkv_heads_from_separate_tensors_device_operation_types.hpp"
#include "create_qkv_heads_from_separate_tensors_program_factory.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::transformer {

struct CreateQKVHeadsSeparateTensorsDeviceOperation {
    using operation_attributes_t =
        ttnn::operations::experimental::create_qkv_heads_from_separate_tensors::operation_attributes_t;
    using tensor_args_t = ttnn::operations::experimental::create_qkv_heads_from_separate_tensors::tensor_args_t;
    using spec_return_value_t =
        ttnn::operations::experimental::create_qkv_heads_from_separate_tensors::spec_return_value_t;
    using tensor_return_value_t =
        ttnn::operations::experimental::create_qkv_heads_from_separate_tensors::tensor_return_value_t;
    using program_factory_t = std::variant<ttnn::operations::experimental::create_qkv_heads_from_separate_tensors::
                                               CreateQKVHeadsSeparateTensorsProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::transformer

namespace ttnn::prim {
std::tuple<Tensor, Tensor, Tensor> create_qkv_heads_from_separate_tensors(
    const Tensor& input_tensor,
    const Tensor& input_tensor_kv,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    bool transpose_k_heads,
    const MemoryConfig& output_mem_config,
    const std::optional<std::array<Tensor, 3>>& optional_output_tensors);
}  // namespace ttnn::prim
