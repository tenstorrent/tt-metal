// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "create_qkv_heads_device_operation.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::transformer {

void CreateQKVHeadsDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_FATAL(input_tensor.layout() == Layout::TILE, "Error");
    TT_FATAL(input_tensor.is_sharded(), "Operands to TM must be sharded");
    const auto input_shape = input_tensor.padded_shape();
    TT_FATAL(input_shape[1] == 1, "Unsupported input shape");

    auto bbox = input_tensor.shard_spec().value().grid.bounding_box();
    TT_FATAL(
        (bbox.end_coord.x < input_tensor.device()->compute_with_storage_grid_size().x &&
         bbox.end_coord.y < input_tensor.device()->compute_with_storage_grid_size().y),
        "Error");
    TT_FATAL(input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED, "Error");
    ShardOrientation shard_orientation = input_tensor.shard_spec().value().orientation;
    bool rm = shard_orientation == ShardOrientation::ROW_MAJOR;
    uint32_t num_h_cores = rm ? bbox.end_coord.y + 1 : bbox.end_coord.x + 1;
    uint32_t num_w_cores = rm ? bbox.end_coord.x + 1 : bbox.end_coord.y + 1;

    TT_FATAL(
        this->num_q_heads % this->num_kv_heads == 0,
        "Number of q heads {} must fit evenly into number of kv heads {}",
        this->num_q_heads,
        this->num_kv_heads);
    TT_FATAL(
        input_shape[3] % (num_w_cores * tt::constants::TILE_WIDTH) == 0,
        "Flattened hidden dimension {} must be a multiple of width cores {} * tile width {} to ensure that each core "
        "gets an even amount of tiles",
        input_shape[3],
        num_w_cores,
        tt::constants::TILE_WIDTH);

    TT_FATAL(this->output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
    TT_FATAL(input_shape[0] == num_h_cores, "Batch size {} must be equal to num cores {}", input_shape[0], num_h_cores);
}

std::vector<ttnn::TensorSpec> CreateQKVHeadsDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.padded_shape();

    const auto q_shape = ttnn::Shape{input_shape[0], this->num_q_heads, input_shape[2], this->head_dim};
    const auto v_shape = ttnn::Shape{input_shape[0], this->num_kv_heads, input_shape[2], this->head_dim};
    const auto k_shape =
        this->transpose_k_heads ? ttnn::Shape{input_shape[0], this->num_kv_heads, head_dim, input_shape[2]} : v_shape;

    if (output_tensors.size() == 3) {
        return {output_tensors[0]->tensor_spec(), output_tensors[1]->tensor_spec(), output_tensors[2]->tensor_spec()};
    }
    // no create_output_tensors variant that takes in optional input tensors?

    CoreRangeSet all_cores = input_tensor.shard_spec().value().grid;
    ShardOrientation shard_orientation = input_tensor.shard_spec().value().orientation;
    auto bbox = all_cores.bounding_box();
    // TODO: Do we need to know cores along row and col?
    // bool rm = shard_orientation == ShardOrientation::ROW_MAJOR;
    // uint32_t num_h_cores = rm ? bbox.end_coord.y + 1 : bbox.end_coord.x + 1;
    // uint32_t num_w_cores = rm ? bbox.end_coord.x + 1 : bbox.end_coord.y + 1;
    uint32_t num_cores = bbox.size();

    // TODO: Do we need?
    // uint32_t num_kv_heads_per_shard = k_shape[1] / num_w_cores;
    // uint32_t num_q_heads_per_shard = q_shape[1] / num_w_cores;

    uint32_t q_shard_h =
        q_shape[0] * q_shape[1] * q_shape[2] / num_cores;  // want the API to work for different sequence lengths
    uint32_t k_shard_h =
        k_shape[0] * k_shape[1] * k_shape[2] / num_cores;  // want the API to work for different sequence lengths
    uint32_t v_shard_h =
        v_shape[0] * v_shape[1] * v_shape[2] / num_cores;  // want the API to work for different sequence lengths

    auto q_spec = tt::tt_metal::ShardSpec(all_cores, {q_shard_h, q_shape[-1]}, shard_orientation);
    auto k_spec = tt::tt_metal::ShardSpec(all_cores, {k_shard_h, k_shape[-1]}, shard_orientation);
    auto v_spec = tt::tt_metal::ShardSpec(all_cores, {v_shard_h, v_shape[-1]}, shard_orientation);
    // create sharded tensors
    auto mem_config_q = this->output_mem_config.with_shard_spec(q_spec);
    auto mem_config_k = this->output_mem_config.with_shard_spec(k_spec);
    auto mem_config_v = this->output_mem_config.with_shard_spec(v_spec);

    TensorSpec out_tensor_q(
        q_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), mem_config_q));
    TensorSpec out_tensor_k(
        k_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), mem_config_k));
    TensorSpec out_tensor_v(
        v_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), mem_config_v));
    return {out_tensor_q, out_tensor_k, out_tensor_v};
}

std::vector<Tensor> CreateQKVHeadsDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    auto specs = compute_output_specs(input_tensors, output_tensors);
    return {
        create_device_tensor(specs[0], input_tensors.at(0).device()),
        create_device_tensor(specs[1], input_tensors.at(0).device()),
        create_device_tensor(specs[2], input_tensors.at(0).device()),
    };
}

tt::tt_metal::operation::ProgramWithCallbacks CreateQKVHeadsDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    return multi_core_create_qkv_heads_sharded(
        input_tensor,
        this->num_q_heads,
        this->num_kv_heads,
        this->head_dim,
        this->transpose_k_heads,
        output_tensors,
        compute_with_storage_grid_size);
}

}  // namespace ttnn::operations::experimental::transformer
