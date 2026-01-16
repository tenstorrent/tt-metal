// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "create_qkv_heads_program_factory.hpp"
#include "ttnn/decorators.hpp"
#include "create_qkv_heads_device_operation_types.hpp"

#include "ttnn/operation.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental::create_qkv_heads {

struct CreateQKVHeadsDeviceOperation {
    using operation_attributes_t = ttnn::operations::experimental::create_qkv_heads::operation_attributes_t;
    using tensor_args_t = ttnn::operations::experimental::create_qkv_heads::tensor_args_t;
    using spec_return_value_t = ttnn::operations::experimental::create_qkv_heads::spec_return_value_t;
    using tensor_return_value_t = ttnn::operations::experimental::create_qkv_heads::tensor_return_value_t;
    using program_factory_t =
        std::variant<ttnn::operations::experimental::create_qkv_heads::program::CreateQKVHeadsProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::create_qkv_heads

namespace ttnn::prim {
ttnn::operations::experimental::create_qkv_heads::tensor_return_value_t create_qkv_heads(
    const Tensor& input_tensor,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    bool transpose_k_heads,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::tuple<Tensor, Tensor, Tensor>>& preallocated_outputs);
}  // namespace ttnn::prim
