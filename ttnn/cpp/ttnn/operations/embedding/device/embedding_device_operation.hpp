// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"

#include "embedding_device_operation_types.hpp"
#include "embeddings_fused_program_factory.hpp"
#include "embeddings_fused_runtime_rows_program_factory.hpp"
#include "embeddings_rm_program_factory.hpp"
#include "embeddings_tilized_indices_program_factory.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct EmbeddingsDeviceOperation {
    using operation_attributes_t = EmbeddingParams;
    using tensor_args_t = EmbeddingInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<
        EmbeddingsFusedProgramFactory,
        EmbeddingsRMProgramFactory,
        EmbeddingsTilizedIndicesProgramFactory,
        EmbeddingsFusedRuntimeRowsProgramFactory
    >;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Runtime-rows factory: attributes + index dtype/layout/batch + table width and dtype; the index
    // length and the table row count are runtime. Everything else: the framework's default hash.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

Tensor embedding(
    const Tensor& input_tensor_arg,
    const Tensor& weight_arg,
    bool tilized,
    EmbeddingsType embeddings_type,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<uint32_t>& pad_token = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    bool runtime_rows = false);

}  // namespace ttnn::prim
