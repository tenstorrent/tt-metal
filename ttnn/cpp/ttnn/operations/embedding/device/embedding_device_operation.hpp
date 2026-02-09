// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

#include "embedding_device_operation_types.hpp"
#include "embeddings_fused_program_factory.hpp"
#include "embeddings_rm_program_factory.hpp"
#include "embeddings_tilized_indices_program_factory.hpp"

namespace ttnn::operations::embedding {

struct EmbeddingsDeviceOperation {
    using operation_attributes_t = EmbeddingParams;
    using tensor_args_t = EmbeddingInputs;
    using spec_return_value_t = embedding::spec_return_value_t;
    using tensor_return_value_t = embedding::tensor_return_value_t;
    using program_factory_t = std::variant<
        program::EmbeddingsFusedProgramFactory,
        program::EmbeddingsRMProgramFactory,
        program::EmbeddingsTilizedIndicesProgramFactory
    >;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

}  // namespace ttnn::operations::embedding

namespace ttnn::prim {
ttnn::operations::embedding::EmbeddingsDeviceOperation::tensor_return_value_t embedding(
    const Tensor& input_tensor_arg,
    const Tensor& weight_arg,
    bool tilized,
    ttnn::operations::embedding::EmbeddingsType embeddings_type,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<uint32_t>& pad_token = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);
}  // namespace ttnn::prim
