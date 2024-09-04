// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "create_qkv_heads_from_separate_tensors_device_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"

namespace ttnn::operations::experimental::transformer {

void CreateQKVHeadsSeparateTensorsDeviceOperation::validate(const std::vector<Tensor> &input_tensors) const {
    using namespace tt::constants;
    const auto& q_input_tensor = input_tensors.at(0);
    const auto& kv_input_tensor = input_tensors.at(1);

    TT_FATAL(q_input_tensor.storage_type() == StorageType::DEVICE && kv_input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(q_input_tensor.buffer() != nullptr && kv_input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(q_input_tensor.get_dtype() == tt::tt_metal::DataType::FLOAT32 || q_input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16 || q_input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");
    TT_FATAL(kv_input_tensor.get_dtype() == q_input_tensor.get_dtype(), "Unsupported data format");
    TT_FATAL(q_input_tensor.get_layout() == Layout::TILE && kv_input_tensor.get_layout() == Layout::TILE);
    TT_FATAL(q_input_tensor.is_sharded() && kv_input_tensor.is_sharded(), "Operands to TM must be sharded");


    auto bbox = q_input_tensor.shard_spec().value().grid.bounding_box();
    TT_FATAL((bbox.end_coord.x < q_input_tensor.device()->compute_with_storage_grid_size().x && bbox.end_coord.y < q_input_tensor.device()->compute_with_storage_grid_size().y));

    TT_FATAL(q_input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);
    TT_FATAL(kv_input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);

    ShardOrientation shard_orientation = q_input_tensor.shard_spec().value().orientation;
    bool rm = shard_orientation == ShardOrientation::ROW_MAJOR;
    uint32_t num_h_cores = rm ? bbox.end_coord.y + 1 : bbox.end_coord.x + 1;
    uint32_t num_w_cores = rm ? bbox.end_coord.x + 1 : bbox.end_coord.y + 1;

    TT_FATAL(this->num_q_heads % num_w_cores == 0, fmt::format("Number of q heads {} must fit evenly into cores {}", this->num_q_heads, num_w_cores));
    TT_FATAL(this->num_kv_heads % num_w_cores == 0, fmt::format("Number of kv heads {} must fit evenly into cores {}", this->num_kv_heads, num_w_cores));

    const auto q_input_shape = q_input_tensor.get_legacy_shape();
    const auto kv_input_shape = kv_input_tensor.get_legacy_shape();
    TT_FATAL(q_input_shape[1] == 1 && kv_input_shape[1] == 1, "Unsupported input shape");
    TT_FATAL(q_input_shape[0] == kv_input_shape[0], fmt::format("Q {} and KV {} batch size must match", q_input_shape[0], kv_input_shape[0]));

    TT_FATAL(q_input_shape[3] % (num_w_cores * TILE_WIDTH) == 0, fmt::format("Flattened hidden dimension {} must be a multiple of width cores {} * tile width {} to ensure that each core gets an even amount of tiles", q_input_shape[3], num_w_cores, TILE_WIDTH));
    TT_FATAL(q_input_shape[0]*q_input_shape[2] % (num_h_cores * TILE_HEIGHT) == 0, fmt::format("Batch {} * Seq Len {} must be a multiple of height cores {} * tile height {} to ensure that each core gets an even amount of tiles", q_input_shape[0], q_input_shape[2], num_h_cores, TILE_HEIGHT));

    TT_FATAL(kv_input_shape[3] % (num_w_cores * TILE_WIDTH) == 0, fmt::format("Flattened hidden dimension {} must be a multiple of width cores {} * tile width {} to ensure that each core gets an even amount of tiles", kv_input_shape[3], num_w_cores, TILE_WIDTH));
    TT_FATAL(kv_input_shape[0]*kv_input_shape[2] % (num_h_cores * TILE_HEIGHT) == 0, fmt::format("Batch {} * Seq Len {} must be a multiple of height cores {} * tile height {} to ensure that each core gets an even amount of tiles", kv_input_shape[0], kv_input_shape[2], num_h_cores, TILE_HEIGHT));

    TT_FATAL((q_input_shape[3]/(this->num_q_heads)) == (kv_input_shape[3]/(2 * this->num_kv_heads)), fmt::format("Head dims must be equal in size! Q {} num_heads {} KV {} num_heads {}", q_input_shape[3], num_q_heads, kv_input_shape[3], num_kv_heads));

    uint32_t q_shard_wt = (q_input_shape[3]) / (num_w_cores * TILE_WIDTH); // number of tiles in width dimension  - multiple tiles per head, multiple heads per group, multiple tensors in group, multiple groups per cores
    uint32_t q_shard_ht = ((q_input_shape[0] * q_input_shape[2]) / (num_w_cores * TILE_HEIGHT));
    uint32_t k_shard_wt = (kv_input_shape[3] / (2 * num_w_cores * TILE_WIDTH));
    uint32_t k_shard_ht = ((kv_input_shape[0] * kv_input_shape[2]) / (num_h_cores * TILE_HEIGHT));

    TT_FATAL(q_shard_ht > 0, "0 height shards on Q");
    TT_FATAL(q_shard_wt > 0, "0 width shards on Q");
    TT_FATAL(k_shard_ht > 0, "0 height shards on K");
    TT_FATAL(k_shard_wt > 0, "0 width shards on K");

    uint32_t per_core_q_tiles = q_shard_ht * q_shard_wt;
    uint32_t per_core_k_tiles = k_shard_ht * k_shard_wt;

    const uint32_t single_tile_size = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(q_input_tensor.get_dtype()));
    TT_FATAL(L1_SIZE >= 2 * (per_core_q_tiles + 2*per_core_k_tiles) * single_tile_size, fmt::format("Workload exceeds L1 capacity"));

    // TODO: Add this back when output is HEIGHT sharded only!
    // TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
    TT_FATAL(q_input_shape[0] == num_h_cores, fmt::format("Batch size {} must be equal to num cores {}", q_input_shape[0], num_h_cores));
}

std::vector<tt::tt_metal::Shape> CreateQKVHeadsSeparateTensorsDeviceOperation::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& input_tensor_kv = input_tensors.at(1);
    const auto input_shape = input_tensor.get_legacy_shape();
    const auto input_shape_kv = input_tensor_kv.get_legacy_shape();

    const auto q_output_shape = {input_shape[0], this->num_q_heads, input_shape[2], this->head_dim};
    const auto v_output_shape = {input_shape_kv[0], this->num_kv_heads, input_shape_kv[2], this->head_dim};
    const auto k_output_shape = this->transpose_k_heads
                                     ? tt::tt_metal::Shape{input_shape_kv[0], this->num_kv_heads, head_dim, input_shape_kv[2]}
                                     : v_output_shape;
    return {q_output_shape, k_output_shape, v_output_shape};
}

operation::ProgramWithCallbacks CreateQKVHeadsSeparateTensorsDeviceOperation::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_q = input_tensors.at(0);
    const auto& input_tensor_kv = input_tensors.at(1);
    CoreCoord compute_with_storage_grid_size = input_tensor_q.device()->compute_with_storage_grid_size();
    return  multi_core_create_q_and_kv_heads_sharded(input_tensor_q, input_tensor_kv, this->num_q_heads, this->num_kv_heads, this->head_dim, this->transpose_k_heads, output_tensors, compute_with_storage_grid_size);
}

std::vector<Tensor> CreateQKVHeadsSeparateTensorsDeviceOperation::create_output_tensors(const std::vector<Tensor>& input_tensors) const  {
    // no create_output_tensors variant that takes in optional input tensors?
    const auto& input_tensor = input_tensors.at(0);

    CoreRangeSet all_cores = input_tensor.shard_spec().value().grid;
    ShardOrientation shard_orientation = input_tensor.shard_spec().value().orientation;
    auto bbox = all_cores.bounding_box();
    // TODO: Do we need to know cores along row and col?
    //bool rm = shard_orientation == ShardOrientation::ROW_MAJOR;
    //uint32_t num_h_cores = rm ? bbox.end_coord.y + 1 : bbox.end_coord.x + 1;
    //uint32_t num_w_cores = rm ? bbox.end_coord.x + 1 : bbox.end_coord.y + 1;
    uint32_t num_cores = bbox.size();

    auto shapes = compute_output_shapes(input_tensors);
    auto q_shape = shapes.at(0);
    auto k_shape = shapes.at(1);
    auto v_shape = shapes.at(2);

    // TODO: Do we need?
    //uint32_t num_kv_heads_per_shard = k_shape[1] / num_w_cores;
    //uint32_t num_q_heads_per_shard = q_shape[1] / num_w_cores;

    uint32_t q_shard_h = q_shape[0] * q_shape[1] * q_shape[2] / num_cores; // want the API to work for different sequence lengths
    uint32_t k_shard_h = k_shape[0] * k_shape[1] * k_shape[2] / num_cores; // want the API to work for different sequence lengths
    uint32_t v_shard_h = v_shape[0] * v_shape[1] * v_shape[2] / num_cores; // want the API to work for different sequence lengths

    auto q_spec = ShardSpec(all_cores, {q_shard_h, q_shape[-1]}, shard_orientation);
    auto k_spec = ShardSpec(all_cores, {k_shard_h, k_shape[-1]}, shard_orientation);
    auto v_spec = ShardSpec(all_cores, {v_shard_h, v_shape[-1]}, shard_orientation);
    // create sharded tensors
    auto mem_config_q = this->output_mem_config;
    mem_config_q.shard_spec = q_spec;

    auto mem_config_k = this->output_mem_config;
    mem_config_k.shard_spec = k_spec;

    auto mem_config_v = this->output_mem_config;
    mem_config_v.shard_spec = v_spec;

    auto out_tensor_q = create_device_tensor(q_shape, input_tensor.get_dtype(), Layout::TILE, input_tensor.device(), mem_config_q);
    auto out_tensor_k = create_device_tensor(k_shape, input_tensor.get_dtype(), Layout::TILE, input_tensor.device(), mem_config_k);
    auto out_tensor_v = create_device_tensor(v_shape, input_tensor.get_dtype(), Layout::TILE, input_tensor.device(), mem_config_v);
    return {out_tensor_q, out_tensor_k, out_tensor_v};
}
}  // namespace ttnn::operations::experimental::transformer
