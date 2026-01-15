// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_device_operation_types.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_program_factory.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::transformer::rotary_embedding {

struct RotaryEmbeddingDeviceOperation {
    using operation_attributes_t = RotaryEmbeddingParams;
    using tensor_args_t = RotaryEmbeddingInputs;
    using spec_return_value_t = rotary_embedding::spec_return_value_t;
    using tensor_return_value_t = rotary_embedding::tensor_return_value_t;
    using program_factory_t = std::variant<rotary_embedding::program::RotaryEmbeddingProgramFactory>;
    using shared_variables_t = rotary_embedding::program::RotaryEmbeddingProgramFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::transformer::rotary_embedding

namespace ttnn::prim {
ttnn::operations::experimental::transformer::rotary_embedding::tensor_return_value_t rotary_embedding(
    const Tensor& input,
    const Tensor& cos,
    const Tensor& sin,
    uint32_t seq_len,
    std::optional<uint32_t> token_idx,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config);
}  // namespace ttnn::prim
