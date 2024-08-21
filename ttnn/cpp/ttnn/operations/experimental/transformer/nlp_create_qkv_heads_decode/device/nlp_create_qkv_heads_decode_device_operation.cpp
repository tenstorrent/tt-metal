// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_decode_device_operation.hpp"
#include "ttnn/operations/core/work_split/work_split.hpp"

#include "tt_metal/host_api.hpp"

namespace ttnn::operations::experimental::transformer {

// Generic NLP CreateHeads op for decode
void NLPCreateHeadsDecodeDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();
    // TODO: Rewrite validation for this decode case
    // NOTE: Checks for head_dim and shape[3] is done in nlp_create_qkv_heads because it's needed to infer head_dim
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.get_dtype() == tt::tt_metal::DataType::FLOAT32 || input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16, "Unsupported data format");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE);

    // input
    TT_FATAL(input_shape[3] % TILE_WIDTH == 0, "Unsupported input shape");  // head_dim must be multiple of TILE_WIDTH
    TT_FATAL(input_shape[2] == 32, "Unsupported input shape");  // 32 users
    TT_FATAL(input_shape[1] == 1, "Unsupported input shape");
    TT_FATAL(input_shape[0] == 1, "Unsupported input shape");
    TT_FATAL(input_tensor.is_sharded(), "Input must be sharded");
    TT_FATAL(input_tensor.shard_spec().value().shape[0] == input_tensor.volume() / input_tensor.get_legacy_shape()[-1]);
    TT_FATAL(input_tensor.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR);
    // we either put everything in one shard or split it into minimum tile width accross as many cores as possible
    TT_FATAL(input_tensor.shard_spec().value().shape[1] == (this->num_q_heads + this->num_kv_heads * 2) * this->head_dim || input_tensor.shard_spec().value().shape[1] == 32);
    auto core_grid = input_tensor.device()->compute_with_storage_grid_size();

    // output
    TT_FATAL(this->output_mem_config.is_sharded() && this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
    uint32_t num_cores = core_grid.x * core_grid.y;
    // Support maximum 32 heads for now
    TT_FATAL(this->num_q_heads <= 32);
    // 1 User Per Core Max and 32 users for now
    TT_FATAL(num_cores >= 32, "Need at least 32 cores for decode");
    TT_FATAL(this->num_q_heads >= this->num_kv_heads);
}

std::vector<tt::tt_metal::Shape> NLPCreateHeadsDecodeDeviceOperation::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    std::vector<Shape> output_shape_vec;
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();

    auto batch = input_tensor.get_shape()[2];
    auto head_dim = this->head_dim;

    // pad up to nearest multiple of TILE_HEIGHT for num_q_heads and num_kv_heads
    auto num_q_heads_padded = (this->num_q_heads / TILE_HEIGHT + 1) * TILE_HEIGHT;
    auto num_kv_heads_padded = (this->num_kv_heads / TILE_HEIGHT + 1) * TILE_HEIGHT;

    const tt::tt_metal::Shape q_output_shape = {input_shape[0], batch, num_q_heads_padded, head_dim};
    const tt::tt_metal::Shape v_output_shape = {input_shape[0], batch, num_kv_heads_padded, head_dim};
    const tt::tt_metal::Shape k_output_shape = v_output_shape;
    return {q_output_shape, k_output_shape, v_output_shape};

}

std::vector<Tensor> NLPCreateHeadsDecodeDeviceOperation::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();
    auto output_shapes = this->compute_output_shapes(input_tensors);
    const auto& q_output_shape = output_shapes[0];

    auto batch = q_output_shape[1];
    auto num_q_heads_padded = (this->num_q_heads / TILE_HEIGHT + 1) * TILE_HEIGHT;
    auto num_kv_heads_padded = (this->num_kv_heads / TILE_HEIGHT + 1) * TILE_HEIGHT;
    auto core_grid = input_tensor.device()->compute_with_storage_grid_size();
    auto q_shard_grid = ttnn::operations::core::work_split::num_cores_to_corerange_set(batch, core_grid, true);
    ShardSpec q_shard_spec{q_shard_grid, {num_q_heads_padded, this->head_dim}};
    auto q_mem_config = this->output_mem_config;
    q_mem_config.shard_spec = q_shard_spec;
    auto kv_shard_grid = ttnn::operations::core::work_split::num_cores_to_corerange_set(batch, core_grid, true);
    ShardSpec kv_shard_spec{kv_shard_grid, {num_kv_heads_padded, this->head_dim}};
    auto kv_mem_config = this->output_mem_config;
    kv_mem_config.shard_spec = kv_shard_spec;
    return {
        create_device_tensor(output_shapes[0], input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), q_mem_config),
        create_device_tensor(output_shapes[1], input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), kv_mem_config),
        create_device_tensor(output_shapes[2], input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), kv_mem_config)
    };
}

operation::ProgramWithCallbacks NLPCreateHeadsDecodeDeviceOperation::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    return  multi_core_nlp_create_qkv_heads_decode(input_tensor, this->num_q_heads, this->num_kv_heads, this->head_dim, output_tensors, compute_with_storage_grid_size);
}

}  // namespace ttnn::operations::experimental::transformer
