// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concatenate_heads.hpp"


#include "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/nlp_concat_heads_device_operation.hpp"

namespace ttnn::operations::transformer {

struct ConcatenateHeads : public ttnn::operations::experimental::transformer::NLPConcatHeadsDeviceOperation {
    void validate(const std::vector<Tensor>& input_tensors) const {

        const auto& input_tensor = input_tensors.at(0);
        const auto head_size = input_tensor.get_shape()[-1];
        const auto padded_head_size = input_tensor.get_legacy_shape()[-1];

        TT_FATAL(
            head_size % ttnn::types::TILE_SIZE == 0,
                "Head size must be a multiple of {} but was found to be {}. Update the matmul that uses the output of this "
                "operation to include padding in the weights.",
                ttnn::types::TILE_SIZE,
                head_size);

        TT_FATAL(
            padded_head_size - head_size == 0,
            "Head size ({}) cannot have tile padding. Ensure that the head size is not padded.", head_size);

        NLPConcatHeadsDeviceOperation::validate(input_tensors);
    }

    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
        const auto& input_tensor = input_tensors.at(0);
        const ttnn::SimpleShape input_logical_shape = input_tensor.get_logical_shape();
        const ttnn::SimpleShape input_padded_shape = input_tensor.get_padded_shape();

        auto batch_size = input_logical_shape[0];
        auto num_heads = input_logical_shape[1];
        auto sequence_size = input_logical_shape[2];
        auto padded_sequence_size = input_padded_shape[2];
        auto head_size = input_logical_shape[3];
        auto padded_head_size = input_padded_shape[3];

        std::array<uint32_t, 3> intended_output_shape = {batch_size, sequence_size, num_heads * head_size};
        std::array<uint32_t, 3> padded_output_shape = {batch_size, padded_sequence_size, num_heads * padded_head_size};
        return {ttnn::Shape(intended_output_shape, padded_output_shape).value};
    }

    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const {
        const auto& input_tensor = input_tensors.at(0);
        if (this->output_mem_config.is_sharded()) {
            ShardSpec shard_spec = input_tensor.shard_spec().value();
            uint32_t num_cores = shard_spec.num_cores();
            uint32_t heads_per_shard = shard_spec.shape[0] / input_tensor.get_legacy_shape()[-2];
            shard_spec.shape = {shard_spec.shape[0] / heads_per_shard, shard_spec.shape[1] * heads_per_shard};
            auto mem_config = this->output_mem_config;
            mem_config.shard_spec = shard_spec;
            return {create_device_tensor(
                this->compute_output_shapes(input_tensors).at(0),
                input_tensor.get_dtype(),
                Layout::TILE,
                input_tensor.device(),
                mem_config)};
        } else {
            return operation::generic_create_output_tensors(
                *this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
        }
    }
};


ttnn::Tensor ExecuteConcatenateHeads::invoke(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    return operation::run(
        ConcatenateHeads{memory_config.value_or(input_tensor.memory_config())},
        {input_tensor}).at(0);
}

}
