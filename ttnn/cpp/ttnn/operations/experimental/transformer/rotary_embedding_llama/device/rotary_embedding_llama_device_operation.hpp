// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/host_api.hpp>
#include "rotary_embedding_llama_device_operation_types.hpp"
#include "rotary_embedding_llama_multi_core_program_factory.hpp"
#include "rotary_embedding_llama_sharded_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::transformer::rotary_embedding_llama {

struct RotaryEmbeddingLlamaDeviceOperation {
    using operation_attributes_t = RotaryEmbeddingLlamaParams;
    using tensor_args_t = RotaryEmbeddingLlamaInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = tt::tt_metal::Tensor;
    using program_factory_t = std::variant<
        rotary_embedding_llama::program::RotaryEmbeddingLlamaMultiCore,
        rotary_embedding_llama::program::RotaryEmbeddingLlamaMultiCoreSharded>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::transformer::rotary_embedding_llama

namespace ttnn::prim {
tt::tt_metal::Tensor rotary_embedding_llama(
    const tt::tt_metal::Tensor& input_tensor,
    const tt::tt_metal::Tensor& cos_cache,
    const tt::tt_metal::Tensor& sin_cache,
    const tt::tt_metal::Tensor& trans_mat,
    bool is_decode_mode,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config);
}  // namespace ttnn::prim
