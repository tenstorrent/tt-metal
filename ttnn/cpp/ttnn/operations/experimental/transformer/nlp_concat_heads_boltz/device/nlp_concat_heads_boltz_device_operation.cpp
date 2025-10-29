// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_concat_heads_boltz_device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

// Generic NLP ConcatHeadsBoltz op
void NLPConcatHeadsBoltzDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_FATAL(
        input_tensor.layout() == tt::tt_metal::Layout::TILE,
        "Input tensor layout must be TILE but got {}",
        input_tensor.layout());
    if (input_tensor.is_sharded()) {
        TT_FATAL(
            input_tensor.memory_config().memory_layout() != tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
            "Input tensor memory layout must not be WIDTH_SHARDED but got {}",
            input_tensor.memory_config().memory_layout());
        auto shard_spec = input_tensor.shard_spec().value();
        TT_FATAL(
            input_tensor.padded_shape()[1] % (shard_spec.shape[0] / input_tensor.padded_shape()[-2]) == 0, "Error");
        // Allow HEIGHT_SHARDED output memory layout for sharded inputs
        // TT_FATAL(this->output_mem_config.memory_layout() != tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED, "Error");
    } else {
        TT_FATAL(
            this->output_mem_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "Output memory config layout must be INTERLEAVED but got {}",
            this->output_mem_config.memory_layout());
    }
}

std::vector<ttnn::TensorSpec> NLPConcatHeadsBoltzDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& input_shape = input_tensor.logical_shape();

    auto num_heads = input_shape[0];
    auto sequence_length = input_shape[2];
    auto head_dim = input_shape[3];

    auto hidden_dim = num_heads * head_dim;

    Shape output_shape({1, sequence_length, sequence_length, hidden_dim});

    if (this->output_mem_config.is_sharded()) {
        tt::tt_metal::ShardSpec shard_spec = input_tensor.shard_spec().value();
        uint32_t heads_per_shard = shard_spec.shape[0] / input_tensor.padded_shape()[-2];
        shard_spec.shape = {shard_spec.shape[0] / heads_per_shard, shard_spec.shape[1] * heads_per_shard};
        auto mem_config = this->output_mem_config.with_shard_spec(shard_spec);
        return {TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem_config))};
    }

    return {TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), output_mem_config))};
}

tt::tt_metal::operation::ProgramWithCallbacks NLPConcatHeadsBoltzDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    return multi_core_nlp_concat_heads_boltz(input_tensor, output_tensor, compute_with_storage_grid_size);
}

}  // namespace ttnn::operations::experimental::transformer
