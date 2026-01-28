// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/rotary_embedding_llama_fused_qk_device_operation_types.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/rotary_embedding_llama_fused_qk_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingLlamaFusedQKDeviceOperation {
    using operation_attributes_t = RotaryEmbeddingLlamaFusedQkParams;
    using tensor_args_t = RotaryEmbeddingLlamaFusedQkInputs;
    using spec_return_value_t = RotaryEmbeddingLlamaFusedQkResultSpec;
    using tensor_return_value_t = RotaryEmbeddingLlamaFusedQkResult;
    using program_factory_t = std::variant<RotaryEmbeddingLlamaFusedQKProgramFactory>;
    using shared_variables_t = RotaryEmbeddingLlamaFusedQKProgramFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
ttnn::experimental::prim::RotaryEmbeddingLlamaFusedQkResult rotary_embedding_llama_fused_qk(
    const Tensor& q_input_tensor,
    const Tensor& k_input_tensor,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const Tensor& trans_mat,
    const tt::tt_metal::MemoryConfig& q_output_mem_config,
    const tt::tt_metal::MemoryConfig& k_output_mem_config,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    bool row_major_QK);
}  // namespace ttnn::prim
