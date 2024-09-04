// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_concat_heads_device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

// Generic NLP ConcatHeads op
void NLPConcatHeadsDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.get_dtype() == tt::tt_metal::DataType::FLOAT32 || input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16 || input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE);

    if (input_tensor.is_sharded()) {
        TT_FATAL(input_tensor.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED);
        auto shard_spec = input_tensor.shard_spec().value();
        TT_FATAL(shard_spec.shape[1] == input_tensor.get_legacy_shape()[-1]);
        TT_FATAL(shard_spec.shape[0] % input_tensor.get_legacy_shape()[-2] == 0);
        TT_FATAL(input_tensor.get_legacy_shape()[1] % (shard_spec.shape[0] / input_tensor.get_legacy_shape()[-2]) == 0);
        TT_FATAL(this->output_mem_config.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED);
    } else {
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
    }
}

std::vector<tt::tt_metal::Shape> NLPConcatHeadsDeviceOperation::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    std::vector<tt::tt_metal::Shape> output_shape_vec;
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();
    output_shape_vec = {(tt::tt_metal::Shape) {input_shape[0], 1, input_shape[2], input_shape[1] * input_shape[3]}};

    auto num_heads = input_shape[1];
    auto sequence_length = input_shape[2];
    auto head_dim = input_shape[3];

    auto hidden_dim = num_heads * head_dim;

    return {{input_shape[0], 1, sequence_length, hidden_dim}};
}

std::vector<Tensor> NLPConcatHeadsDeviceOperation::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (this->output_mem_config.is_sharded()) {
        ShardSpec shard_spec = input_tensor.shard_spec().value();
        uint32_t num_cores = shard_spec.num_cores();
        uint32_t heads_per_shard = shard_spec.shape[0] / input_tensor.get_legacy_shape()[-2];
        shard_spec.shape = {shard_spec.shape[0] / heads_per_shard, shard_spec.shape[1] * heads_per_shard};
        auto mem_config = this->output_mem_config;
        mem_config.shard_spec = shard_spec;
        return {create_device_tensor(this->compute_output_shapes(input_tensors).at(0), input_tensor.get_dtype(), Layout::TILE, input_tensor.device(), mem_config)};
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks NLPConcatHeadsDeviceOperation::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    return  multi_core_nlp_concat_heads(input_tensor, output_tensor, compute_with_storage_grid_size);
}

}  // namespace ttnn::operations::experimental::transformer
