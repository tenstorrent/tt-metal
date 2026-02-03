// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

#include "embedding_backward_device_operation_types.hpp"
#include "embedding_backward_program_factory.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::prim {

struct EmbeddingBackwardDeviceOperation {
    using operation_attributes_t = EmbeddingBackwardParams;
    using tensor_args_t = EmbeddingBackwardInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<EmbeddingBackwardProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

ttnn::Tensor embedding_backward(
    const Tensor& index_tensor,
    const Tensor& grad_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype,
    uint32_t num_embeddings,
    const std::optional<Tensor>& preallocated_output = std::nullopt);

}  // namespace ttnn::prim
