// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

#include "embedding_backward_device_operation_types.hpp"
#include "embedding_backward_program_factory.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::embedding_backward {

struct EmbeddingBackwardDeviceOperation {
    using operation_attributes_t = embedding_backward::operation_attributes_t;
    using tensor_args_t = embedding_backward::tensor_args_t;
    using spec_return_value_t = embedding_backward::spec_return_value_t;
    using tensor_return_value_t = embedding_backward::tensor_return_value_t;
    using program_factory_t = std::variant<program::EmbeddingBackwardProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& index_tensor,
        const Tensor& grad_tensor,
        const tt::tt_metal::MemoryConfig& output_mem_config,
        const tt::tt_metal::DataType& output_dtype,
        uint32_t num_embeddings,
        const std::optional<Tensor>& preallocated_output);
};

}  // namespace ttnn::operations::embedding_backward

namespace ttnn::prim {

constexpr auto embedding_backward = ttnn::register_operation<
    "ttnn::prim::embedding_backward",
    ttnn::operations::embedding_backward::EmbeddingBackwardDeviceOperation>();

}  // namespace ttnn::prim
