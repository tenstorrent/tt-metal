// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "max_pool2d_device_op.hpp"

/**
 * New maxpool2d implementation that uses the new sliding window infrastructure.
 */

namespace ttnn::operations::pool {

MaxPool2D::program_factory_t MaxPool2D::select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
    return MultiCore{};
}

void validate_maxpool(const Tensor& input, const sliding_window::SlidingWindowConfig& sliding_window_config, const MemoryConfig& out_mem_config) {
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input.buffer() != nullptr , "Operands to reshape need to be allocated in buffers on device!");
    TT_FATAL(input.get_dtype() == DataType::BFLOAT16, "Only BFLOAT16 supported for now");
    TT_FATAL(input.get_layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR supported for now");

    // NOTE: This is not a hard requirement. If need to support non-power-of-2, simply change the address generator in reader to generic one.
    uint32_t in_nbytes_c = (input.get_legacy_shape()[3]) * (input.get_dtype() == DataType::BFLOAT16 ? 2 : 1);
    bool is_pow2 = (in_nbytes_c & (in_nbytes_c - 1)) == 0;
    TT_FATAL(is_pow2, "Row size (nchannels * bytes = {}) should be power of 2 ({}).", in_nbytes_c, is_pow2);

    TT_FATAL(input.memory_config().is_sharded(), "Input needs to be sharded");
    //TT_FATAL(input.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Only height sharded tensors are supported.");

    TT_FATAL(out_mem_config.is_sharded(), "Output memory config needs to be sharded");
    TT_FATAL(out_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Only height sharded tensors are supported.");
}

void MaxPool2D::validate_on_program_cache_miss(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    return validate_maxpool(tensors.input_tensor_, op_attr.sliding_window_config_, op_attr.memory_config_);
}

void MaxPool2D::validate_on_program_cache_hit(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    return validate_maxpool(tensors.input_tensor_, op_attr.sliding_window_config_, op_attr.memory_config_);
}

MaxPool2D::shape_return_value_t MaxPool2D::compute_output_shapes(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    auto& input = tensors.input_tensor_;
    auto& sliding_window_config = op_attr.sliding_window_config_;
    auto& out_mem_config = op_attr.memory_config_;
    auto& output_dtype = op_attr.output_dtype_;

    // NOTE: Only for RM
    // NOTE2: Assuming { N, 1, H * W, C }
    // NOTE3: Assuming output data type is same as input
    const auto input_shape = input.get_legacy_shape();

    // confirm that the output size supplied to the function matches
    uint32_t out_h = sliding_window_config.get_output_shape()[1];
    uint32_t out_w = sliding_window_config.get_output_shape()[2];

    bool is_out_tiled = output_dtype == DataType::BFLOAT8_B;

    // need to pad the last dim to TILE_WIDTH
    uint32_t out_c = input_shape[3];
    uint32_t out_c_padded = ceil_multiple_of(out_c, (out_c <= 16) ? 16 : tt::constants::TILE_WIDTH);
    uint32_t out_pagesize = out_c_padded * datum_size(datatype_to_dataformat_converter(input.get_dtype()));
    uint32_t out_nhw = sliding_window_config.batch_size * out_h * out_w;

    uint32_t out_nhw_padded = tt::round_up(out_nhw, (is_out_tiled ? tt::constants::TILE_HEIGHT : 1) * sliding_window_config.num_cores_nhw);

    // {1, 1, N * H * W, C}
    const auto out_dims = std::vector<uint32_t>({1, 1, out_nhw_padded, out_c_padded});
    const auto padding = Padding(
        {{0, 0}, {0, 0}, {0, out_nhw_padded - out_nhw}, {0, out_c_padded - out_c}},
        Padding::PadValue::NegativeInfinity);
    auto out_shape = Shape(tt::tt_metal::LegacyShape(out_dims, padding));
    return out_shape;
}

MaxPool2D::tensor_return_value_t MaxPool2D::create_output_tensors(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    auto& input = tensors.input_tensor_;
    auto& sliding_window_config = op_attr.sliding_window_config_;
    auto& out_mem_config = op_attr.memory_config_;
    auto& output_dtype = op_attr.output_dtype_;

    Shape output_shape = compute_output_shapes(op_attr, tensors);
    auto mem_config = out_mem_config;
    if (mem_config.shard_spec.has_value()) {
        mem_config.shard_spec->shape[1] = output_shape[3];
    } else {
        uint32_t ncores = input.shard_spec().value().num_cores();
        TT_FATAL(ncores == sliding_window_config.num_cores_nhw, "Number of cores should match");
        uint32_t nbatch = output_shape[0];
        uint32_t out_nhw_padded = output_shape[0] * output_shape[1] * output_shape[2];
        uint32_t out_nhw_per_core = out_nhw_padded / ncores;
        CoreRangeSet shard_grid = sliding_window_config.core_range_set;
        std::array<uint32_t, 2> shard_shape = {out_nhw_per_core, input.get_legacy_shape()[-1]};
        mem_config.shard_spec = ShardSpec{shard_grid, shard_shape, ShardOrientation::ROW_MAJOR, false};
    }

    // return create_device_tensor(output_shape, input.get_dtype(), input.get_layout(), input.device(), mem_config);
    return create_device_tensor(output_shape, output_dtype, input.get_layout(), input.device(), mem_config);
}

tt::stl::hash::hash_t MaxPool2D::compute_program_hash(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    auto input_mem_config = tensors.input_tensor_.memory_config();
    auto dtype = tensors.input_tensor_.dtype();
    return operation::hash_operation<MaxPool2D>(op_attr.sliding_window_config_.get_hash(), op_attr.memory_config_, input_mem_config, dtype);
}

operation::OpPerformanceModel MaxPool2D::create_op_performance_model(const operation_attributes_t& op_attr, const tensor_args_t& inputs, const Tensor& output) {
    const auto& input = inputs.input_tensor_;
    const auto& input_shape = input.get_shape();
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
    uint32_t pad_h = sliding_window_config.pad_hw.first;
    uint32_t pad_w = sliding_window_config.pad_hw.second;

    // GS specific parameters
    int num_cores = 9 * 12;
    int tensix_mul_adds_per_cycle_lofi = 2048;

    // Calculate output dimensions: relevant for window/stride based OPs (conv, maxpool, downsample)
    int output_height = std::floor((activation_h - filter_h + 2 * pad_h) / stride_h + 1);
    int output_width = std::floor((activation_w - filter_w + 2 * pad_w) / stride_w + 1);

    // Calculate number of mul/add / compare operations
    int64_t num_mul_adds_per_elem = activation_c * filter_h * filter_w; // 1 multiply and 1 add per element
    int64_t num_mul_adds = num_mul_adds_per_elem * output_height * output_width * output_channels * batch_size;

    int ideal_dev_clock_cycles = std::ceil((float)num_mul_adds / (float)(num_cores * tensix_mul_adds_per_cycle_lofi));

    operation::OpPerformanceModel result({input}, {output}, ideal_dev_clock_cycles);
    return result;
}


std::tuple<MaxPool2D::operation_attributes_t, MaxPool2D::tensor_args_t> MaxPool2D::invoke(
    const Tensor& input_tensor,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    DataType output_dtype,
    MemoryConfig memory_config) {
    return {
        operation_attributes_t{sliding_window_config, output_dtype, memory_config},
        tensor_args_t{input_tensor}
    };
}

} // namespace ttnn::operations::pool
