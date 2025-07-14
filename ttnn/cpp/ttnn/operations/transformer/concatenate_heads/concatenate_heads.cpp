// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concatenate_heads.hpp"

#include "ttnn/operations/experimental/transformer/nlp_concat_heads/device/nlp_concat_heads_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::transformer {

struct ConcatenateHeads : public ttnn::operations::experimental::transformer::NLPConcatHeadsDeviceOperation {
    void validate(const std::vector<Tensor>& input_tensors) const {
        const auto& input_tensor = input_tensors.at(0);
        const auto& input_logical_shape = input_tensor.logical_shape();
        const auto head_size = input_logical_shape[-1];
        const auto padded_head_size = input_tensor.padded_shape()[-1];

        TT_FATAL(input_logical_shape.rank() == 4, "Input tensor must have rank 4. Shape: {}", input_logical_shape);

        TT_FATAL(
            head_size % ttnn::types::TILE_SIZE == 0,
            "Head size must be a multiple of {} but was found to be {}. Update the matmul that uses the output of this "
            "operation to include padding in the weights.",
            ttnn::types::TILE_SIZE,
            head_size);

        TT_FATAL(
            padded_head_size - head_size == 0,
            "Head size ({}) cannot have tile padding. Ensure that the head size is not padded.",
            head_size);

        NLPConcatHeadsDeviceOperation::validate(input_tensors);
    }

    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const {
        const auto& input_tensor = input_tensors.at(0);
        const ttnn::Shape& input_logical_shape = input_tensor.logical_shape();
        const ttnn::Shape& input_padded_shape = input_tensor.padded_shape();

        auto batch_size = input_logical_shape[0];
        auto num_heads = input_logical_shape[1];
        auto sequence_size = input_logical_shape[2];
        auto padded_sequence_size = input_padded_shape[2];
        auto head_size = input_logical_shape[3];
        auto padded_head_size = input_padded_shape[3];

        Shape intended_output_shape({batch_size, sequence_size, num_heads * head_size});
        Shape padded_output_shape({batch_size, padded_sequence_size, num_heads * padded_head_size});

        if (this->output_mem_config.is_sharded()) {
            ShardSpec shard_spec = input_tensor.shard_spec().value();
            uint32_t heads_per_shard = shard_spec.shape[0] / input_tensor.padded_shape()[-2];
            shard_spec.shape = {shard_spec.shape[0] / heads_per_shard, shard_spec.shape[1] * heads_per_shard};
            auto mem_config = this->output_mem_config.with_shard_spec(shard_spec);
            return {TensorSpec(
                intended_output_shape,
                TensorLayout::fromPaddedShape(
                    input_tensor.dtype(),
                    PageConfig(Layout::TILE),
                    mem_config,
                    intended_output_shape,
                    padded_output_shape))};
        }

        return {TensorSpec(
            intended_output_shape,
            TensorLayout::fromPaddedShape(
                input_tensor.dtype(),
                PageConfig(Layout::TILE),
                output_mem_config,
                intended_output_shape,
                padded_output_shape))};
    }
};

ttnn::Tensor ExecuteConcatenateHeads::invoke(
    const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    return operation::run(ConcatenateHeads{memory_config.value_or(input_tensor.memory_config())}, {input_tensor}).at(0);
}

}  // namespace ttnn::operations::transformer
