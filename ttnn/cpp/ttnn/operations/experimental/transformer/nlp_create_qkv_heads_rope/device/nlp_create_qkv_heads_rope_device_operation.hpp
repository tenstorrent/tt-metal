// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_rope/device/nlp_create_qkv_heads_rope_device_operation_types.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_rope/device/nlp_create_qkv_heads_rope_program_factory.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct NlpCreateQkvHeadsRopeDeviceOperation {
    using operation_attributes_t = NlpCreateQkvHeadsRopeParams;
    using tensor_args_t = NlpCreateQkvHeadsRopeInputs;
    using spec_return_value_t = std::tuple<TensorSpec, TensorSpec, TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor, Tensor>;
    using program_factory_t = std::variant<NlpCreateQkvHeadsRopeProgramFactory>;
    using shared_variables_t = NlpCreateQkvHeadsRopeProgramFactory::shared_variables_t;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
std::tuple<Tensor, Tensor, Tensor> nlp_create_qkv_heads_rope(
    const Tensor& qkv,
    const Tensor& cos,
    const Tensor& sin,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t seq_len,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config);
}  // namespace ttnn::prim
