// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pool_op.hpp"

#include <tt-metalium/math.hpp>
#include <utility>

/**
 * Generic pool implementation that uses the new sliding window infrastructure.
 */

namespace ttnn::operations::pool {

Pool2D::program_factory_t Pool2D::select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
    return MultiCore{};
}

void validate_pool2d(
    const Tensor& input,
    const Pool2DType pool_type,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    const MemoryConfig& out_mem_config) {
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input.buffer() != nullptr, "Operands to reshape need to be allocated in buffers on device!");
    TT_FATAL(input.get_dtype() == DataType::BFLOAT16, "Only BFLOAT16 supported for now");
    TT_FATAL(input.get_layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR supported for now");

    TT_FATAL(input.memory_config().is_sharded(), "Input needs to be sharded");
    TT_FATAL(out_mem_config.is_sharded(), "Output memory config needs to be sharded");

    const auto input_shape = input.get_padded_shape();
    TT_FATAL(
        (input_shape[3] % tt::constants::TILE_WIDTH == 0) || (input_shape[3] == 16),
        "Input channels ({}) should be padded to nearest TILE_WIDTH ({}) or should be 16",
        input_shape[3],
        tt::constants::TILE_WIDTH);

    // check that C dimnenion is a multiple of num_shards_c for all but height sharding
    TensorMemoryLayout in_memory_layout = input.memory_config().memory_layout();
    if (in_memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
        uint32_t num_shards_c = sliding_window_config.num_cores_c;
        TT_FATAL(
            input_shape[3] % num_shards_c == 0,
            "For width and block sharding, input channels ({}) should be divisible by num_shards ({})",
            input_shape[3],
            num_shards_c);
    }

    TT_FATAL(
        sliding_window_config.ceil_mode == false || pool_type == Pool2DType::MAX_POOL2D,
        "Ceil mode set to true not supported for avg pool op");
}

void Pool2D::validate_on_program_cache_miss(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    return validate_pool2d(
        tensors.input_tensor_, op_attr.pool_type_, op_attr.sliding_window_config_, op_attr.memory_config_);
}

void Pool2D::validate_on_program_cache_hit(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    return validate_pool2d(
        tensors.input_tensor_, op_attr.pool_type_, op_attr.sliding_window_config_, op_attr.memory_config_);
}

Pool2D::spec_return_value_t Pool2D::compute_output_specs(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    auto& input = tensors.input_tensor_;
    auto& sliding_window_config = op_attr.sliding_window_config_;
    auto& out_mem_config = op_attr.memory_config_;
    auto& output_dtype = op_attr.output_dtype_;

    // NOTE: Only for RM
    // NOTE2: Assuming { N, 1, H * W, C }
    // NOTE3: Assuming output data type is same as input
    const auto input_shape = input.get_padded_shape();

    // confirm that the output size supplied to the function matches
    uint32_t out_h = sliding_window_config.get_output_shape()[1];
    uint32_t out_w = sliding_window_config.get_output_shape()[2];

    bool is_out_tiled = output_dtype == DataType::BFLOAT8_B;

    // need to pad the last dim to TILE_WIDTH
    uint32_t out_c = input_shape[3];
    uint32_t out_c_padded = tt::round_up(out_c, (out_c <= 16) ? 16 : tt::constants::TILE_WIDTH);
    uint32_t out_nhw = sliding_window_config.batch_size * out_h * out_w;

    uint32_t out_nhw_padded =
        tt::round_up(out_nhw, (is_out_tiled ? tt::constants::TILE_HEIGHT : 1) * sliding_window_config.num_cores_nhw);

    // {1, 1, N * H * W, C}
    const ttnn::Shape padded_output_shape({1, 1, out_nhw_padded, out_c_padded});
    const ttnn::Shape output_shape({1, 1, out_nhw, out_c});

    auto mem_config = out_mem_config;
    if (mem_config.shard_spec().has_value()) {
        auto shard_spec = mem_config.shard_spec().value();
        shard_spec.shape[1] = input.shard_spec().value().shape[1];
        mem_config = mem_config.with_shard_spec(shard_spec);
    } else {
        uint32_t ncores = input.shard_spec().value().num_cores();
        TT_FATAL(ncores == sliding_window_config.num_cores_nhw, "Number of cores should match");
        uint32_t out_nhw_per_core = output_shape[0] * output_shape[1] * output_shape[2] / ncores;
        CoreRangeSet shard_grid = sliding_window_config.core_range_set;
        std::array<uint32_t, 2> shard_shape = {out_nhw_per_core, input.get_padded_shape()[-1]};
        mem_config =
            mem_config.with_shard_spec(tt::tt_metal::ShardSpec{shard_grid, shard_shape, ShardOrientation::ROW_MAJOR});
    }

    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout::fromPaddedShape(
            output_dtype, tt::tt_metal::PageConfig(input.get_layout()), mem_config, output_shape, padded_output_shape));
}

Pool2D::tensor_return_value_t Pool2D::create_output_tensors(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    auto output_spec = compute_output_specs(op_attr, tensors);
    return create_device_tensor(output_spec, tensors.input_tensor_.device());
}

tt::stl::hash::hash_t Pool2D::compute_program_hash(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    auto input_mem_config = tensors.input_tensor_.memory_config();
    auto dtype = tensors.input_tensor_.dtype();
    return tt::tt_metal::operation::hash_operation<Pool2D>(
        op_attr.sliding_window_config_.get_hash(), op_attr.pool_type_, op_attr.memory_config_, input_mem_config, dtype);
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Pool2D::tensor_return_value_t> Pool2D::create_op_performance_model(
    const operation_attributes_t& op_attr, const tensor_args_t& inputs, const Tensor& output) {
    const auto& input = inputs.input_tensor_;
    const auto& input_shape = input.get_logical_shape();
    auto sliding_window_config = op_attr.sliding_window_config_;
    uint32_t batch_size = sliding_window_config.batch_size;
    uint32_t activation_h = sliding_window_config.input_hw.first;
    uint32_t activation_w = sliding_window_config.input_hw.second;
    uint32_t activation_c = input_shape[3];
    uint32_t output_channels = input_shape[3];

    uint32_t filter_h = sliding_window_config.window_hw.first;
    uint32_t filter_w = sliding_window_config.window_hw.second;
    uint32_t stride_h = sliding_window_config.stride_hw.first;
    uint32_t stride_w = sliding_window_config.stride_hw.second;
    uint32_t pad_h = sliding_window_config.get_pad_h();
    uint32_t pad_w = sliding_window_config.get_pad_w();

    // GS specific parameters
    int num_cores = 9 * 12;
    int tensix_mul_adds_per_cycle_lofi = 2048;

    // Calculate output dimensions: relevant for window/stride based OPs (conv, pool, downsample)
    int output_height = std::floor((activation_h - filter_h + pad_h) / stride_h + 1);
    int output_width = std::floor((activation_w - filter_w + pad_w) / stride_w + 1);

    // Calculate number of mul/add / compare operations
    int64_t num_mul_adds_per_elem = activation_c * filter_h * filter_w;  // 1 multiply and 1 add per element
    int64_t num_mul_adds = num_mul_adds_per_elem * output_height * output_width * output_channels * batch_size;

    int ideal_dev_clock_cycles = std::ceil((float)num_mul_adds / (float)(num_cores * tensix_mul_adds_per_cycle_lofi));

    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input}, {output}, ideal_dev_clock_cycles);
    return result;
}

std::tuple<Pool2D::operation_attributes_t, Pool2D::tensor_args_t> Pool2D::invoke(
    const Tensor& input_tensor,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    Pool2DType pool_type,
    DataType output_dtype,
    MemoryConfig memory_config) {
    return {
        operation_attributes_t{
            .sliding_window_config_ = sliding_window_config,
            .pool_type_ = pool_type,
            .output_dtype_ = output_dtype,
            .memory_config_ = std::move(memory_config)},
        tensor_args_t{input_tensor}};
}

}  // namespace ttnn::operations::pool
