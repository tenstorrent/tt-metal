// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_concat_heads_decode_device_operation.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::transformer {

// NLP ConcatHeads op for decode
void NLPConcatHeadsDecodeDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_padded_shape();

    // input tensor and shape
    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.get_dtype() == tt::tt_metal::DataType::FLOAT32 ||
            input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16,
        "Unsupported data format");
    TT_FATAL(input_tensor.get_layout() == tt::tt_metal::Layout::TILE, "Error");
    TT_FATAL(input_shape[0] == 1, "seqlen=1 for decode");
    TT_FATAL(input_shape[1] <= 32, "currently only support less than 32 users");
    TT_FATAL(input_shape[2] == 32, "currently only support 32 padded heads");
    TT_FATAL(input_shape[2] >= this->num_heads, "head_dim must be multiple of TILE_WIDTH");

    // input tensor shard spec
    TT_FATAL(input_tensor.is_sharded(), "Error");
    TT_FATAL(input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED, "Error");
    auto shard_spec = input_tensor.shard_spec().value();
    TT_FATAL(shard_spec.shape[1] == input_tensor.get_padded_shape()[-1], "Error");
    TT_FATAL(shard_spec.shape[0] == input_tensor.get_padded_shape()[-2], "Error");
    auto num_cores = shard_spec.grid.num_cores();
    TT_FATAL(num_cores == input_shape[1], "num_cores must be equal to num users");
    if (this->on_subcoregrids) {
        TT_FATAL(num_cores >= this->num_heads, "For subcoregrid inputs, we only support num_heads<=num_cores");
    }
}

std::vector<ttnn::TensorSpec> NLPConcatHeadsDecodeDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_padded_shape();

    auto num_heads = this->num_heads;
    auto sequence_length = input_shape[0];
    auto batch = input_shape[1];
    auto head_dim = input_shape[3];
    // pad batch to 32 if necessary
    if (batch < 32) {
        batch = 32;
    }

    auto hidden_dim = num_heads * head_dim;

    Shape output_shape({sequence_length, 1, batch, hidden_dim});

    CoreRangeSet output_core_grid;
    if (this->on_subcoregrids) {
        const auto input_core_ranges = input_tensor.shard_spec().value().grid.ranges();
        CoreRangeSet input_core_grid = input_tensor.shard_spec().value().grid;
        const auto start_coord = input_core_ranges[0].start_coord;
        output_core_grid =
            tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(start_coord, num_heads, input_core_grid, true);
    } else {
        output_core_grid = tt::tt_metal::num_cores_to_corerangeset(
            num_heads, input_tensor.device()->compute_with_storage_grid_size(), true);
    }

    tt::tt_metal::ShardSpec shard_spec{output_core_grid, {batch, head_dim}};
    auto mem_config = tt::tt_metal::MemoryConfig{
        tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED, tt::tt_metal::BufferType::L1, shard_spec};

    return {TensorSpec(
        output_shape, tt::tt_metal::TensorLayout(input_tensor.get_dtype(), tt::tt_metal::Layout::TILE, mem_config))};
}

tt::tt_metal::operation::ProgramWithCallbacks NLPConcatHeadsDecodeDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    if (this->on_subcoregrids) {
        return multi_core_nlp_concat_heads_decode_subcoregrids(
            input_tensor, output_tensor, compute_with_storage_grid_size);
    }
    return multi_core_nlp_concat_heads_decode(input_tensor, output_tensor, compute_with_storage_grid_size);
}

}  // namespace ttnn::operations::experimental::transformer
