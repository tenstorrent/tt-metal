// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_concat_heads_decode_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"

namespace ttnn::operations::experimental::transformer {

// NLP ConcatHeads op for decode
void NLPConcatHeadsDecodeDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();

    // input tensor and shape
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.get_dtype() == tt::tt_metal::DataType::FLOAT32 || input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16, "Unsupported data format");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE, "Error");
    TT_FATAL(input_shape[0] == 1, "seqlen=1 for decode");
    TT_FATAL(input_shape[1] <= 32, "currently only support less than 32 users");
    TT_FATAL(input_shape[2] == 32, "currently only support 32 padded heads");
    TT_FATAL(input_shape[2] >= this->num_heads, "head_dim must be multiple of TILE_WIDTH");

    // input tensor shard spec
    TT_FATAL(input_tensor.is_sharded(), "Error");
    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
    auto shard_spec = input_tensor.shard_spec().value();
    TT_FATAL(shard_spec.shape[1] == input_tensor.get_legacy_shape()[-1], "Error");
    TT_FATAL(shard_spec.shape[0] == input_tensor.get_legacy_shape()[-2], "Error");
    auto shard_grid = shard_spec.grid.bounding_box().grid_size();
    auto num_cores = shard_grid.x * shard_grid.y;
    TT_FATAL(num_cores == input_shape[1], "num_cores must be equal to num users");
}

std::vector<tt::tt_metal::Shape> NLPConcatHeadsDecodeDeviceOperation::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();

    auto num_heads = this->num_heads;
    auto sequence_length = input_shape[0];
    auto batch = input_shape[1];
    auto head_dim = input_shape[3];
    // pad batch to 32 if necessary
    if (batch < 32) {
        batch = 32;
    }

    auto hidden_dim = num_heads * head_dim;

    return {{sequence_length, 1, batch, hidden_dim}};
}

std::vector<Tensor> NLPConcatHeadsDecodeDeviceOperation::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto num_heads = this->num_heads;
    const auto input_shape = input_tensor.get_legacy_shape();
    auto sequence_length = input_shape[0];
    auto head_dim = input_shape[3];
    auto output_shape = this->compute_output_shapes(input_tensors).at(0);
    auto batch = output_shape[2];

    auto core_grid = DeviceComputeWithStorageGridSize(input_tensor.device());
    auto shard_grid = num_cores_to_corerange_set(num_heads, core_grid, true);
    ShardSpec shard_spec{shard_grid, {batch, head_dim}};
    auto mem_config = tt::tt_metal::MemoryConfig{TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1};
    mem_config.shard_spec = shard_spec;

    return {create_device_tensor(output_shape, input_tensor.get_dtype(), Layout::TILE, input_tensor.device(), mem_config)};
}

operation::ProgramWithCallbacks NLPConcatHeadsDecodeDeviceOperation::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    CoreCoord compute_with_storage_grid_size = DeviceComputeWithStorageGridSize(input_tensor.device());

    return  multi_core_nlp_concat_heads_decode(input_tensor, output_tensor, compute_with_storage_grid_size);
}


}  // namespace ttnn::operations::experimental::transformer
