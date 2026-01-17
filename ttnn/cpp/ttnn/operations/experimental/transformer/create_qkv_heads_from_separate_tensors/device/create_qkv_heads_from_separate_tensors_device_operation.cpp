// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "create_qkv_heads_from_separate_tensors_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

using namespace ttnn::operations::experimental::create_qkv_heads_from_separate_tensors;

void CreateQKVHeadsSeparateTensorsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& q_input_tensor = tensor_args.input_tensor;
    const auto& kv_input_tensor = tensor_args.input_tensor_kv;
    using namespace tt::constants;

    TT_FATAL(
        q_input_tensor.storage_type() == StorageType::DEVICE && kv_input_tensor.storage_type() == StorageType::DEVICE,
        "Operands to TM need to be on device!");
    TT_FATAL(
        q_input_tensor.buffer() != nullptr && kv_input_tensor.buffer() != nullptr,
        "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        q_input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32 ||
            q_input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            q_input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_FATAL(kv_input_tensor.dtype() == q_input_tensor.dtype(), "Unsupported data format");
    TT_FATAL(
        q_input_tensor.layout() == Layout::TILE && kv_input_tensor.layout() == Layout::TILE,
        "Q and KV input tensors must have TILE layout but got Q: {}, KV: {}",
        q_input_tensor.layout(),
        kv_input_tensor.layout());
    TT_FATAL(q_input_tensor.is_sharded() && kv_input_tensor.is_sharded(), "Operands to TM must be sharded");

    auto bbox = q_input_tensor.shard_spec().value().grid.bounding_box();
    TT_FATAL(
        (bbox.end_coord.x < q_input_tensor.device()->compute_with_storage_grid_size().x &&
         bbox.end_coord.y < q_input_tensor.device()->compute_with_storage_grid_size().y),
        "Error");

    TT_FATAL(
        q_input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED,
        "Q input tensor memory layout must be BLOCK_SHARDED but got {}",
        q_input_tensor.memory_config().memory_layout());
    TT_FATAL(
        kv_input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED,
        "KV input tensor memory layout must be BLOCK_SHARDED but got {}",
        kv_input_tensor.memory_config().memory_layout());

    ShardOrientation shard_orientation = q_input_tensor.shard_spec().value().orientation;
    bool rm = shard_orientation == ShardOrientation::ROW_MAJOR;
    uint32_t num_h_cores = rm ? bbox.end_coord.y + 1 : bbox.end_coord.x + 1;
    uint32_t num_w_cores = rm ? bbox.end_coord.x + 1 : bbox.end_coord.y + 1;

    TT_FATAL(
        operation_attributes.num_q_heads % num_w_cores == 0,
        "Number of q heads {} must fit evenly into cores {}",
        operation_attributes.num_q_heads,
        num_w_cores);
    TT_FATAL(
        operation_attributes.num_kv_heads % num_w_cores == 0,
        "Number of kv heads {} must fit evenly into cores {}",
        operation_attributes.num_kv_heads,
        num_w_cores);

    const auto& q_input_shape = q_input_tensor.padded_shape();
    const auto& kv_input_shape = kv_input_tensor.padded_shape();
    TT_FATAL(q_input_shape[1] == 1 && kv_input_shape[1] == 1, "Unsupported input shape");
    TT_FATAL(
        q_input_shape[0] == kv_input_shape[0],
        "Q {} and KV {} batch size must match",
        q_input_shape[0],
        kv_input_shape[0]);

    TT_FATAL(
        q_input_shape[3] % (num_w_cores * TILE_WIDTH) == 0,
        "Flattened hidden dimension {} must be a multiple of width cores {} * tile width {} to ensure that each core "
        "gets an even amount of tiles",
        q_input_shape[3],
        num_w_cores,
        TILE_WIDTH);
    TT_FATAL(
        q_input_shape[0] * q_input_shape[2] % (num_h_cores * TILE_HEIGHT) == 0,
        "Batch {} * Seq Len {} must be a multiple of height cores {} * tile height {} to ensure that each core gets an "
        "even amount of tiles",
        q_input_shape[0],
        q_input_shape[2],
        num_h_cores,
        TILE_HEIGHT);

    TT_FATAL(
        kv_input_shape[3] % (num_w_cores * TILE_WIDTH) == 0,
        "Flattened hidden dimension {} must be a multiple of width cores {} * tile width {} to ensure that each core "
        "gets an even amount of tiles",
        kv_input_shape[3],
        num_w_cores,
        TILE_WIDTH);
    TT_FATAL(
        kv_input_shape[0] * kv_input_shape[2] % (num_h_cores * TILE_HEIGHT) == 0,
        "Batch {} * Seq Len {} must be a multiple of height cores {} * tile height {} to ensure that each core gets an "
        "even amount of tiles",
        kv_input_shape[0],
        kv_input_shape[2],
        num_h_cores,
        TILE_HEIGHT);

    TT_FATAL(
        (q_input_shape[3] / (operation_attributes.num_q_heads)) ==
            (kv_input_shape[3] / (2 * operation_attributes.num_kv_heads)),
        "Head dims must be equal in size! Q {} num_heads {} KV {} num_heads {}",
        q_input_shape[3],
        operation_attributes.num_q_heads,
        kv_input_shape[3],
        operation_attributes.num_kv_heads);

    uint32_t q_shard_wt =
        (q_input_shape[3]) /
        (num_w_cores * TILE_WIDTH);  // number of tiles in width dimension  - multiple tiles per head, multiple heads
                                     // per group, multiple tensors in group, multiple groups per cores
    uint32_t q_shard_ht = ((q_input_shape[0] * q_input_shape[2]) / (num_w_cores * TILE_HEIGHT));
    uint32_t k_shard_wt = (kv_input_shape[3] / (2 * num_w_cores * TILE_WIDTH));
    uint32_t k_shard_ht = ((kv_input_shape[0] * kv_input_shape[2]) / (num_h_cores * TILE_HEIGHT));

    TT_FATAL(q_shard_ht > 0, "0 height shards on Q");
    TT_FATAL(q_shard_wt > 0, "0 width shards on Q");
    TT_FATAL(k_shard_ht > 0, "0 height shards on K");
    TT_FATAL(k_shard_wt > 0, "0 width shards on K");

    uint32_t per_core_q_tiles = q_shard_ht * q_shard_wt;
    uint32_t per_core_k_tiles = k_shard_ht * k_shard_wt;

    const uint32_t l1_size = q_input_tensor.device()->l1_size_per_core();
    const uint32_t single_tile_size =
        tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(q_input_tensor.dtype()));
    TT_FATAL(
        l1_size >= 2 * (per_core_q_tiles + 2 * per_core_k_tiles) * single_tile_size, "Workload exceeds L1 capacity");

    // TODO: Add this back when output is HEIGHT sharded only!
    // TT_FATAL(this->output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
    TT_FATAL(
        q_input_shape[0] == num_h_cores, "Batch size {} must be equal to num cores {}", q_input_shape[0], num_h_cores);
}

void CreateQKVHeadsSeparateTensorsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

CreateQKVHeadsSeparateTensorsDeviceOperation::spec_return_value_t
CreateQKVHeadsSeparateTensorsDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_tensor_kv = tensor_args.input_tensor_kv;
    const auto& input_shape = input_tensor.padded_shape();
    const auto& input_shape_kv = input_tensor_kv.padded_shape();

    Shape q_shape({input_shape[0], operation_attributes.num_q_heads, input_shape[2], operation_attributes.head_dim});
    Shape v_shape(
        {input_shape_kv[0], operation_attributes.num_kv_heads, input_shape_kv[2], operation_attributes.head_dim});
    const auto k_shape = operation_attributes.transpose_k_heads ? Shape(
                                                                      {input_shape_kv[0],
                                                                       operation_attributes.num_kv_heads,
                                                                       operation_attributes.head_dim,
                                                                       input_shape_kv[2]})
                                                                : v_shape;

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
    auto mem_config_q = operation_attributes.output_mem_config.with_shard_spec(q_spec);
    auto mem_config_k = operation_attributes.output_mem_config.with_shard_spec(k_spec);
    auto mem_config_v = operation_attributes.output_mem_config.with_shard_spec(v_spec);

    auto out_tensor_q = TensorSpec(
        q_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), mem_config_q));
    auto out_tensor_k = TensorSpec(
        k_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), mem_config_k));
    auto out_tensor_v = TensorSpec(
        v_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), mem_config_v));
    return std::make_tuple(out_tensor_q, out_tensor_k, out_tensor_v);
}

CreateQKVHeadsSeparateTensorsDeviceOperation::tensor_return_value_t
CreateQKVHeadsSeparateTensorsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& optional_output_tensors = tensor_args.optional_output_tensors;
    if (optional_output_tensors.has_value()) {
        return std::make_tuple(
            optional_output_tensors.value()[0], optional_output_tensors.value()[1], optional_output_tensors.value()[2]);
    }
    // Create tensors from specs
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    return std::make_tuple(
        create_device_tensor(std::get<0>(output_specs), tensor_args.input_tensor.device()),
        create_device_tensor(std::get<1>(output_specs), tensor_args.input_tensor.device()),
        create_device_tensor(std::get<2>(output_specs), tensor_args.input_tensor.device()));
}

CreateQKVHeadsSeparateTensorsDeviceOperation::program_factory_t
CreateQKVHeadsSeparateTensorsDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return CreateQKVHeadsSeparateTensorsProgramFactory{};
}

tt::stl::hash::hash_t CreateQKVHeadsSeparateTensorsDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return tt::tt_metal::operation::hash_operation<CreateQKVHeadsSeparateTensorsDeviceOperation>(
        operation_attributes.num_q_heads,
        operation_attributes.num_kv_heads,
        operation_attributes.head_dim,
        operation_attributes.transpose_k_heads,
        program_factory.index(),
        tensor_args);
}

}  // namespace ttnn::operations::experimental::transformer

namespace ttnn::prim {

std::tuple<Tensor, Tensor, Tensor> create_qkv_heads_from_separate_tensors(
    const Tensor& input_tensor,
    const Tensor& input_tensor_kv,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    bool transpose_k_heads,
    const MemoryConfig& output_mem_config,
    const std::optional<std::array<Tensor, 3>>& optional_output_tensors) {
    using OperationType = ttnn::operations::experimental::transformer::CreateQKVHeadsSeparateTensorsDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .num_q_heads = num_q_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .transpose_k_heads = transpose_k_heads,
        .output_mem_config = output_mem_config};
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input_tensor,
        .input_tensor_kv = input_tensor_kv,
        .optional_output_tensors = optional_output_tensors};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
