// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_halo_program_factory.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/sliding_window/halo/device/halo_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <array>

namespace ttnn::operations::sliding_window::halo {

using namespace tt::tt_metal;

thread_local std::unordered_map<std::size_t, std::uint32_t>
    HaloDeviceOperation::sliding_window_max_out_nsticks_per_core = {};

// TODO: Look into increasing this to tradeoff some L1 for performance (#19980)
HaloDeviceOperation::program_factory_t HaloDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return data_movement::program::UntilizeWithHaloProgramFactory{};
}

void HaloDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    // validate input data tensor
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        // skip the untilize, only do halo
        log_debug(tt::LogOp, "Input is ROW_MAJOR, no need to untilize.");
    } else {
        TT_FATAL(
            input_tensor.physical_volume() % tt::constants::TILE_HW == 0,
            "Input tensor physical volume ({}) must be divisible by TILE_HW ({})",
            input_tensor.physical_volume(),
            tt::constants::TILE_HW);
    }
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Only height, width or block sharded tensors are supported.");
    TT_FATAL(input_tensor.shard_spec().has_value(), "Shard spec should not be empty");
}

void HaloDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

HaloDeviceOperation::spec_return_value_t HaloDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.padded_shape();
    ttnn::Shape output_shape = ttnn::Shape(input_shape.to_array_4D());

    uint32_t nbatch = input_shape[0];
    uint32_t total_nsticks = args.config.num_cores_nhw * args.max_out_nsticks_per_core;

    // output_shape[0] remains same
    // output_shape[1] remains same
    // output_shape[2] changes
    // output_shape[3] remains same
    output_shape[2] = (uint32_t)std::ceil((float)total_nsticks / nbatch);

    log_debug(
        tt::LogOp, "output_shape: [{} {} {} {}]", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    log_debug(tt::LogOp, "max_out_nsticks_per_core: {}", args.max_out_nsticks_per_core);
    log_debug(
        tt::LogOp, "size : {}", args.in_nsticks_per_core * input_tensor.memory_config().shard_spec()->shape[1] * 2);
    log_debug(tt::LogOp, "num_cores_nhw: {}", args.config.num_cores_nhw);

    tt::tt_metal::DataType output_dtype;
    switch (input_tensor.dtype()) {
        case tt::tt_metal::DataType::FLOAT32: output_dtype = tt::tt_metal::DataType::FLOAT32; break;
        case tt::tt_metal::DataType::UINT16: output_dtype = tt::tt_metal::DataType::UINT16; break;
        default: output_dtype = tt::tt_metal::DataType::BFLOAT16; break;
    }

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == args.output_memory_config.memory_layout(),
        "{} {}",
        input_tensor.memory_config(),
        args.output_memory_config);

    if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        auto input_core_range = *(input_tensor.memory_config().shard_spec()->grid.ranges().begin());
        auto output_core_range = *(args.output_memory_config.shard_spec()->grid.ranges().begin());
        auto input_core_w = input_core_range.end_coord.y - input_core_range.start_coord.y + 1;
        auto output_core_w = output_core_range.end_coord.y - output_core_range.start_coord.y + 1;

        TT_FATAL(
            input_core_w == output_core_w, "Input core width {} != Output core width {}", input_core_w, output_core_w);
    }

    std::array<uint32_t, 2> shard_shape = {
        tt::div_up(output_shape[0] * output_shape[2], args.config.num_cores_nhw),
        input_tensor.memory_config().shard_spec()->shape[1]};

    auto out_mem_config = args.output_memory_config.with_shard_spec(ShardSpec{
        args.output_memory_config.shard_spec()->grid,
        shard_shape,
        args.output_memory_config.shard_spec()->orientation});
    auto padded_output_shape = output_shape;
    padded_output_shape[-2] = tt::round_up(padded_output_shape[-2], shard_shape[0]);
    padded_output_shape[-1] = tt::round_up(padded_output_shape[-1], shard_shape[1]);
    return TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            output_dtype, PageConfig(Layout::ROW_MAJOR), out_mem_config, output_shape, padded_output_shape));
}

HaloDeviceOperation::tensor_return_value_t HaloDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

}  // namespace ttnn::operations::sliding_window::halo

namespace ttnn::prim {
ttnn::operations::sliding_window::halo::HaloDeviceOperation::tensor_return_value_t halo(
    const Tensor& input_tensor,
    const ttnn::operations::sliding_window::SlidingWindowConfig& config,
    uint32_t pad_val,
    bool remote_read,
    bool transpose_mcast,
    const MemoryConfig& output_memory_config,
    bool is_out_tiled,
    bool config_tensors_in_dram) {
    using OperationType = ttnn::operations::sliding_window::halo::HaloDeviceOperation;

    TT_FATAL(input_tensor.memory_config().is_sharded(), "Halo expects sharded input tensor");
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Only height, width or block sharded tensors are supported.");
    // NOTE: for HEIGHT_SHARDED, ncores_nhw == ncores
    //       for BLOCK_SHARDED, ncores_nhw is just the ncores along height dim (last tensor dim is split along
    //       width)
    auto sliding_window_hash = config.get_hash();
    if (!OperationType::sliding_window_max_out_nsticks_per_core.contains(sliding_window_hash)) {
        auto op_trace_metadata = ttnn::operations::sliding_window::generate_op_trace_metadata(config);
        auto shard_boundaries = ttnn::operations::sliding_window::generate_shard_boundaries(config);
        OperationType::sliding_window_max_out_nsticks_per_core.emplace(
            sliding_window_hash, ttnn::operations::sliding_window::generate_max_out_nsticks_per_core(shard_boundaries));
    }

    uint32_t max_out_nsticks_per_core = OperationType::sliding_window_max_out_nsticks_per_core.at(sliding_window_hash);
    uint32_t in_nsticks_per_core = input_tensor.memory_config().shard_spec()->shape[0];
    ttnn::operations::sliding_window::ParallelConfig p_config;
    p_config.grid = input_tensor.shard_spec().value().grid;
    p_config.shard_scheme = input_tensor.memory_config().memory_layout();
    p_config.shard_orientation = input_tensor.shard_spec().value().orientation;

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .config = config,
            .parallel_config = p_config,
            .pad_val = pad_val,
            .remote_read = remote_read,
            .transpose_mcast = transpose_mcast,
            .max_out_nsticks_per_core = max_out_nsticks_per_core,
            .in_nsticks_per_core = in_nsticks_per_core,
            .output_memory_config = output_memory_config,
            .is_out_tiled = is_out_tiled,
            .config_tensors_in_dram = config_tensors_in_dram},
        OperationType::tensor_args_t{
            .input_tensor = input_tensor,
        });
}
}  // namespace ttnn::prim
