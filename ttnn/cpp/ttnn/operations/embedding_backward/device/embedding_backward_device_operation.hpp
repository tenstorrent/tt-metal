// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

#include "embedding_backward_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/core.hpp"

namespace ttnn::prim {

struct EmbeddingBackwardDeviceOperation {
    using operation_attributes_t = EmbeddingBackwardParams;
    using tensor_args_t = EmbeddingBackwardInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

// True when the output-stationary (one-hot matmul) program is used for these inputs: a bfloat16 gradient whose index
// scratch fits in L1. It accumulates in FP32 DEST and writes each output tile once, for a bfloat16 output only: the FPU matmul accumulate
// is not accurate enough for the FLOAT32 output contract, which stays on the scatter path.
bool embedding_backward_uses_onehot(const Tensor& index_tensor, const Tensor& grad_tensor, uint32_t num_embeddings);

ttnn::Tensor embedding_backward(
    const Tensor& index_tensor,
    const Tensor& grad_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype,
    uint32_t num_embeddings,
    const std::optional<Tensor>& preallocated_output = std::nullopt);

}  // namespace ttnn::prim
