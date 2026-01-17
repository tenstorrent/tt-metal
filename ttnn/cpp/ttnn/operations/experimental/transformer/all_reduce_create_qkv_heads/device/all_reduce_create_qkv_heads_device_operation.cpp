// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_create_qkv_heads_device_operation.hpp"

#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::ccl::all_reduce_create_qkv_heads {

constexpr int MAX_HEAD = 32;

AllReduceCreateQkvHeadsDeviceOperation::program_factory_t
AllReduceCreateQkvHeadsDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return program::AllReduceCreateQkvHeadsMeshWorkloadFactory{};
}

void AllReduceCreateQkvHeadsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void AllReduceCreateQkvHeadsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& buffer_tensor = tensor_args.buffer_tensor;

    // Lambda for repeated tensor validation checks (lesson from layernorm PR)
    auto validate_tensor_on_device = [](const Tensor& tensor, const std::string& tensor_name) {
        TT_FATAL(
            tensor.storage_type() == StorageType::DEVICE,
            "Operand {} needs to be on device, but has storage_type {}",
            tensor_name,
            tensor.storage_type());
        TT_FATAL(
            tensor.buffer() != nullptr,
            "Operand {} needs to be allocated in buffers on device, but buffer is null",
            tensor_name);
    };

    // Validate input tensors
    validate_tensor_on_device(input_tensor, "input_tensor");
    validate_tensor_on_device(buffer_tensor, "buffer_tensor");

    const auto& page_size = input_tensor.buffer()->page_size();
    TT_FATAL(
        page_size % input_tensor.buffer()->alignment() == 0,
        "All Gather currently requires aligned pages. page_size={}, alignment={}",
        page_size,
        input_tensor.buffer()->alignment());

    TT_FATAL(
        operation_attributes.ring_size % 2 == 0,
        "AllReduceAsync currently only supports even number of blocks in the reduction kernel. ring_size={}",
        operation_attributes.ring_size);

    TT_FATAL(
        operation_attributes.num_links > 0,
        "Error, num_links should be more than 0 but has {}",
        operation_attributes.num_links);
    TT_FATAL(
        operation_attributes.num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelized over rows. num_links={}, grid_size.y={}",
        operation_attributes.num_links,
        input_tensor.device()->compute_with_storage_grid_size().y);

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout for input tensor {}.",
        input_tensor.memory_config().memory_layout());

    TT_FATAL(
        buffer_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout for buffer tensor {}.",
        buffer_tensor.memory_config().memory_layout());

    TT_FATAL(
        operation_attributes.all_reduce_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout for output tensor {}.",
        operation_attributes.all_reduce_mem_config.memory_layout());

    TT_FATAL(
        buffer_tensor.memory_config().shard_spec()->grid.contains(
            operation_attributes.all_reduce_mem_config.shard_spec()->grid),
        "The output tensor must reside on a subset of the cores of the buffer tensor");

    const uint32_t output_shard_shape_volume = operation_attributes.all_reduce_mem_config.shard_spec()->shape[0] *
                                               operation_attributes.all_reduce_mem_config.shard_spec()->shape[1];
    const uint32_t buffer_shard_shape_volume =
        buffer_tensor.memory_config().shard_spec()->shape[0] * buffer_tensor.memory_config().shard_spec()->shape[1];
    TT_FATAL(
        output_shard_shape_volume * operation_attributes.ring_size <= buffer_shard_shape_volume,
        "The shard size for the buffer must be large enough to hold the intermediate tensor. Require at least {} but "
        "has {}",
        output_shard_shape_volume * operation_attributes.ring_size,
        buffer_shard_shape_volume);

    // Validate for create qkv heads
    const auto& input_shape = input_tensor.logical_shape();

    // TODO: Rewrite validation for this decode case
    // NOTE: Checks for head_dim and shape[3] is done in nlp_create_qkv_heads because it's needed to infer head_dim
    TT_FATAL(
        operation_attributes.dtype == tt::tt_metal::DataType::BFLOAT16,
        "Unsupported data format {}, currently only bfloat16 is supported",
        operation_attributes.dtype);
    TT_FATAL(
        input_tensor.layout() == Layout::TILE,
        "Only tile layout is supported for input tensor, but got {}",
        input_tensor.layout());

    // Input shape validation
    const uint32_t num_users_supported = 32;
    uint32_t num_users = input_shape[2];
    TT_FATAL(
        input_shape[3] % tt::constants::TILE_WIDTH == 0,
        "Unsupported input shape = {}, input_shape[3]={} must be multiple of TILE_WIDTH={}",
        input_shape,
        input_shape[3],
        tt::constants::TILE_WIDTH);  // head_dim must be multiple of TILE_WIDTH
    TT_FATAL(
        num_users <= num_users_supported,
        "Unsupported input shape = {}, num_users={} exceeds max supported={}",
        input_shape,
        num_users,
        num_users_supported);  // 32 users
    TT_FATAL(
        input_shape[1] == 1, "Unsupported input shape = {}, input_shape[1]={} must be 1", input_shape, input_shape[1]);
    TT_FATAL(
        input_shape[0] == 1, "Unsupported input shape = {}, input_shape[0]={} must be 1", input_shape, input_shape[0]);

    const auto& QKV_memcfg = input_tensor.memory_config();
    if (input_tensor.is_sharded()) {
        TT_FATAL(
            QKV_memcfg.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
            "Current input memory layout is {}. It must be width sharded",
            QKV_memcfg.memory_layout());
        TT_FATAL(
            input_tensor.shard_spec().value().shape[0] ==
                input_tensor.physical_volume() / input_tensor.padded_shape()[-1],
            "Shard shape must be correct");
        TT_FATAL(
            input_tensor.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
            "Shard orientation must be ROW_MAJOR, but got {}",
            input_tensor.shard_spec().value().orientation);

        /* Don't validate batch_offset and slice_size for now, as they will be provided by the user
        TT_FATAL(
            !(batch_offset.has_value() ^ operation_attributes.slice_size.has_value()),
            "Both batch_offset and slice_size must be provided or neither");
        if (batch_offset.has_value() && operation_attributes.slice_size.has_value()) {
            TT_FATAL(batch_offset.value().logical_shape()[0] == 1, "batch_offset must be unary tensor");
            num_users = operation_attributes.slice_size.value();
        }
        */
    }

    // Output validation
    TT_FATAL(
        operation_attributes.final_mem_config.is_sharded() &&
            operation_attributes.final_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Output tensor must be height sharded, but got memory_layout={}",
        operation_attributes.final_mem_config.memory_layout());

    // Support maximum 32 heads for now
    TT_FATAL(
        operation_attributes.num_heads <= MAX_HEAD,
        "There are {} q heads only {} are supported",
        operation_attributes.num_heads,
        MAX_HEAD);
    TT_FATAL(
        operation_attributes.num_heads >= operation_attributes.num_kv_heads,
        "num_q_heads={} must be greater than or equal to num_kv_heads={}",
        operation_attributes.num_heads,
        operation_attributes.num_kv_heads);

    uint32_t num_cores;
    if (operation_attributes.input_on_subcoregrids) {
        auto input_core_grid = input_tensor.shard_spec().value().grid;
        num_cores = input_core_grid.num_cores();
    } else {
        auto core_grid_size = input_tensor.device()->compute_with_storage_grid_size();
        num_cores = core_grid_size.x * core_grid_size.y;
    }
    // 1 User Per Core Max and 32 users for now
    TT_FATAL(
        num_cores >= 2 * num_users,
        "Input coregrid size is {}. Need cores atleast double of num_users={} for decode when q and k heads are not "
        "overlapping coregrid",
        num_cores,
        num_users);
}

AllReduceCreateQkvHeadsDeviceOperation::spec_return_value_t
AllReduceCreateQkvHeadsDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.logical_shape();

    tt::tt_metal::TensorLayout output_tensor_layout = tt::tt_metal::TensorLayout(
        operation_attributes.dtype,
        input_tensor.tensor_spec().page_config(),
        operation_attributes.all_reduce_mem_config);
    auto all_reduce_tensor_spec = TensorSpec(input_shape, output_tensor_layout);

    auto batch = input_shape[2];
    if (operation_attributes.slice_size.has_value()) {
        batch = operation_attributes.slice_size.value();
    }

    auto head_dim = operation_attributes.head_dim;

    const Shape q_output_shape({input_shape[0], batch, operation_attributes.num_heads, head_dim});
    const Shape v_output_shape({input_shape[0], batch, operation_attributes.num_kv_heads, head_dim});
    const Shape& k_output_shape = v_output_shape;

    auto num_q_heads_padded =
        ((operation_attributes.num_heads - 1) / tt::constants::TILE_HEIGHT + 1) * tt::constants::TILE_HEIGHT;
    auto num_kv_heads_padded =
        ((operation_attributes.num_heads - 1) / tt::constants::TILE_HEIGHT + 1) * tt::constants::TILE_HEIGHT;

    CoreRangeSet q_shard_grid, k_shard_grid, v_shard_grid;
    auto sub_core_grid = operation_attributes.final_mem_config.shard_spec()->grid;
    auto start_core_coord = sub_core_grid.bounding_box().start_coord;
    auto next_core_coord = start_core_coord;

    q_shard_grid =
        tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(start_core_coord, batch, sub_core_grid, true);

    CoreRangeSet q_batch_grid =
        tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(start_core_coord, batch + 1, sub_core_grid, true);
    if (!q_batch_grid.ranges().empty()) {
        next_core_coord = q_batch_grid.ranges().back().end_coord;
    }
    k_shard_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(next_core_coord, batch, sub_core_grid, true);

    CoreRangeSet q_two_batch_grid =
        tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(start_core_coord, (2 * batch) + 1, sub_core_grid, true);
    if (!q_two_batch_grid.ranges().empty()) {
        next_core_coord = q_two_batch_grid.ranges().back().end_coord;
    }
    v_shard_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(next_core_coord, batch, sub_core_grid, true);

    tt::tt_metal::ShardSpec q_shard_spec{q_shard_grid, {num_q_heads_padded, operation_attributes.head_dim}};
    tt::tt_metal::ShardSpec k_shard_spec{k_shard_grid, {num_kv_heads_padded, operation_attributes.head_dim}};
    tt::tt_metal::ShardSpec v_shard_spec{v_shard_grid, {num_kv_heads_padded, operation_attributes.head_dim}};
    MemoryConfig q_mem_config = operation_attributes.final_mem_config.with_shard_spec(q_shard_spec);
    MemoryConfig k_mem_config = operation_attributes.final_mem_config.with_shard_spec(k_shard_spec);
    MemoryConfig v_mem_config = operation_attributes.final_mem_config.with_shard_spec(v_shard_spec);

    return {
        .all_reduce = all_reduce_tensor_spec,
        .q = TensorSpec(
            q_output_shape,
            tt::tt_metal::TensorLayout(
                operation_attributes.dtype, tt::tt_metal::PageConfig(input_tensor.layout()), q_mem_config)),
        .k = TensorSpec(
            k_output_shape,
            tt::tt_metal::TensorLayout(
                operation_attributes.dtype, tt::tt_metal::PageConfig(input_tensor.layout()), k_mem_config)),
        .v = TensorSpec(
            v_output_shape,
            tt::tt_metal::TensorLayout(
                operation_attributes.dtype, tt::tt_metal::PageConfig(input_tensor.layout()), v_mem_config))};
}

AllReduceCreateQkvHeadsDeviceOperation::tensor_return_value_t
AllReduceCreateQkvHeadsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    auto* device = tensor_args.input_tensor.device();

    return {
        .all_reduce = create_device_tensor(output_specs.all_reduce, device),
        .q = create_device_tensor(output_specs.q, device),
        .k = create_device_tensor(output_specs.k, device),
        .v = create_device_tensor(output_specs.v, device)};
}

tt::stl::hash::hash_t AllReduceCreateQkvHeadsDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto input_shape = input_tensor.padded_shape();
    auto input_memory_layout = input_tensor.layout();
    auto input_dtype = input_tensor.dtype();
    auto input_memory_config = input_tensor.memory_config();

    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    // Hash individual fields to avoid hashing non-hashable types like GlobalSemaphore
    return tt::tt_metal::operation::hash_operation<AllReduceCreateQkvHeadsDeviceOperation>(
        operation_attributes.num_links,
        operation_attributes.ring_size,
        operation_attributes.all_reduce_mem_config,
        operation_attributes.topology,
        operation_attributes.cluster_axis,
        program_factory.index(),
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

}  // namespace ttnn::operations::experimental::ccl::all_reduce_create_qkv_heads

namespace ttnn::prim {

ttnn::operations::experimental::ccl::all_reduce_create_qkv_heads::tensor_return_value_t all_reduce_create_qkv_heads(
    const Tensor& input_tensor,
    Tensor& buffer_tensor,
    const Tensor& batch_offset_tensor,
    uint32_t num_links,
    uint32_t ring_size,
    const MemoryConfig& all_reduce_mem_config,
    ttnn::ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    uint32_t head_dim,
    bool use_noc1_only,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    bool input_on_subcoregrids,
    std::optional<uint32_t> slice_size,
    const MemoryConfig& final_mem_config,
    DataType dtype,
    uint32_t cluster_axis) {
    using OperationType =
        ttnn::operations::experimental::ccl::all_reduce_create_qkv_heads::AllReduceCreateQkvHeadsDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t(
        num_links,
        ring_size,
        all_reduce_mem_config,
        topology,
        semaphore,
        sub_device_id,
        head_dim,
        use_noc1_only,
        num_heads,
        num_kv_heads,
        input_on_subcoregrids,
        slice_size,
        final_mem_config,
        dtype,
        cluster_axis);
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input_tensor, .buffer_tensor = buffer_tensor, .batch_offset_tensor = batch_offset_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
