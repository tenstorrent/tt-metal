// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op_width_sharded_program_factory.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <tt-metalium/math.hpp>

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/constants.hpp>

#include <tt-metalium/work_split.hpp>
#include "tt-metalium/shape.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/cb_utils.hpp"

#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/shape/shape.hpp"

namespace ttnn::operations::conv::conv2d {
Conv2dDeviceOperation::program_factory_t Conv2dDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    if (tensor_args.a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        // Use width sharded implementation
        return program::Conv2dWidthShardedProgramFactory{};
    }  // Use regular sharded implementation
    return program::Conv2dShardedProgramFactory{};
}

TensorSpec Conv2dDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& /*tensor_args*/) {
    const auto& input_tensor_a_shape = args.input_tensor_shape;
    uint32_t batch_size = input_tensor_a_shape[0];

    auto sliding_window_output_shape = args.sliding_window_config.get_output_shape();
    uint32_t conv_output_h = sliding_window_output_shape[1];
    uint32_t conv_output_w = sliding_window_output_shape[2];

    // Tiled output shape is padded shape. Padded to tile shape.
    auto shape_w = batch_size * conv_output_h * conv_output_w;
    auto shape_c = args.output_channels;
    auto padded_shape_w = args.parallelization_config.num_cores_nhw *
                          args.parallelization_config.per_core_out_matrix_height_ntile * tt::constants::TILE_HEIGHT;
    auto padded_shape_c = tt::round_up(args.output_channels, tt::constants::TILE_WIDTH);
    ttnn::Shape output_shape({1, 1, shape_w, shape_c});
    ttnn::Shape padded_output_shape({1, 1, padded_shape_w, padded_shape_c});

    auto output_layout = args.untilize_out ? Layout::ROW_MAJOR : Layout::TILE;
    if (args.memory_config.is_sharded()) {
        std::array<uint32_t, 2> shard_shape = {
            args.parallelization_config.per_core_out_matrix_height_ntile * tt::constants::TILE_HEIGHT,
            args.parallelization_config.per_core_out_matrix_width_ntile * tt::constants::TILE_WIDTH};
        auto shard_grid = args.memory_config.shard_spec().value().grid;
        auto shard_spec =
            tt::tt_metal::ShardSpec{shard_grid, shard_shape, args.memory_config.shard_spec().value().orientation};
        auto mem_config = args.memory_config.with_shard_spec(shard_spec);
        return TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                args.dtype,
                tt::tt_metal::PageConfig(output_layout),
                mem_config,
                tt::tt_metal::Alignment(
                    {tt::constants::TILE_HEIGHT,
                     tt::constants::TILE_WIDTH})  // Conv2D always outputs in tile multiples, even if output layout is
                                                  // Row Major.
                ));
    }
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout::fromPaddedShape(
            args.dtype,
            tt::tt_metal::PageConfig(output_layout),
            args.memory_config,
            output_shape,
            padded_output_shape));
}

Tensor Conv2dDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.a.device());
}

void Conv2dDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.a;
    const auto& input_tensor_b = tensor_args.b;
    TT_FATAL(input_tensor_a.memory_config().is_sharded(), "Activation tensor should be sharded.");
    TT_FATAL(!input_tensor_b.memory_config().is_sharded(), "Weights tensor should not be sharded.");
    if (args.untilize_out) {
        TT_FATAL(
            (args.dtype == DataType::BFLOAT16) || (args.dtype == DataType::FLOAT32),
            "Untilize output requires BFLOAT16 or FLOAT32 data type but got {}",
            args.dtype);
    }
    if (args.memory_config.is_sharded()) {
        uint32_t per_core_out_matrix_width_ntiles = args.parallelization_config.per_core_out_matrix_width_ntile;
        uint32_t out_width_ntiles =
            compute_output_specs(args, tensor_args).padded_shape()[-1] / tt::constants::TILE_WIDTH;
        if (args.memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
            TT_FATAL(
                per_core_out_matrix_width_ntiles == out_width_ntiles,
                "Per-core output matrix width in tiles ({}) must equal output width in tiles ({}) for height sharded "
                "layout",
                per_core_out_matrix_width_ntiles,
                out_width_ntiles);
            TT_FATAL(
                args.block_config.out_subblock_w_ntiles == out_width_ntiles ||
                    args.block_config.out_subblock_h_ntiles == 1,
                "Error");
        } else if (args.memory_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            // For block sharded, out_width per core is shard width, and this is split along row
            // TODO: We should clean this up and relax constraints on out_subblock h and w
            if (args.memory_config.shard_spec().value().orientation == ShardOrientation::COL_MAJOR) {
                out_width_ntiles = tt::div_up(out_width_ntiles, args.parallelization_config.grid_size.y);
            } else {
                out_width_ntiles = tt::div_up(out_width_ntiles, args.parallelization_config.grid_size.x);
            }
        }
        TT_FATAL(
            args.block_config.out_subblock_w_ntiles == per_core_out_matrix_width_ntiles ||
                args.block_config.out_subblock_h_ntiles == 1,
            "Error");
    }
}

void Conv2dDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

tt::stl::hash::hash_t Conv2dDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    hashable_operation_attributes_t hashable_args = {
        .sliding_window_config = args.sliding_window_config,
        .output_channels = args.output_channels,
        .untilize_out = args.untilize_out,
        .has_bias = args.has_bias,
        .activation = args.activation,
        .parallelization_config = args.parallelization_config,
        .block_config = args.block_config,
        .memory_config = args.memory_config,
        .dtype = args.dtype,
        .input_tensor_shape = args.input_tensor_shape,
        .compute_kernel_config = args.compute_kernel_config,
        .enable_act_double_buffer = args.enable_act_double_buffer,
        .enable_weights_double_buffer = args.enable_weights_double_buffer,
        .enable_activation_reuse = args.enable_activation_reuse,
        .config_tensors_in_dram = args.config_tensors_in_dram,
        .force_split_reader = args.force_split_reader,
    };
    return tt::stl::hash::hash_objects_with_default_seed(hashable_args, tensor_args);
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> Conv2dDeviceOperation::create_op_performance_model(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    const auto& input_tensor_a_shape = args.input_tensor_shape;
    uint32_t batch_size = input_tensor_a_shape[0];
    uint32_t conv_activation_h = input_tensor_a_shape[1];
    uint32_t conv_activation_w = input_tensor_a_shape[2];
    uint32_t conv_activation_c = input_tensor_a_shape[3];
    uint32_t filter_h = (uint32_t)args.sliding_window_config.window_hw.first;   // filter_h
    uint32_t filter_w = (uint32_t)args.sliding_window_config.window_hw.second;  // filter_W
    uint32_t stride_h = (uint32_t)args.sliding_window_config.stride_hw.first;
    uint32_t stride_w = (uint32_t)args.sliding_window_config.stride_hw.second;
    uint32_t dilation_h = (uint32_t)args.sliding_window_config.dilation_hw.first;
    uint32_t dilation_w = (uint32_t)args.sliding_window_config.dilation_hw.second;

    const CoreCoord compute_grid = output_tensor.device()->compute_with_storage_grid_size();
    const int num_cores = compute_grid.x * compute_grid.y;
    // The Wormhole/Blackhole matrix engine performs 8x16 x 16x16 = 8x16 in a single cycle.
    // This is 2*8*16*16 = 4096 muladds in a single cycle.
    constexpr int tensix_mul_adds_per_cycle_lofi = 4096;

    // Calculate output dimensions: relevant for window/stride based OPs (conv, maxpool, downsample)
    auto [output_height, output_width] = calculate_output_image_size(
        {conv_activation_h, conv_activation_w},
        {filter_h, filter_w},
        {stride_h, stride_w},
        args.sliding_window_config.padding,
        {dilation_h, dilation_w});

    // Calculate number of mul/add operations
    // TODO: add bias modeling
    int64_t num_mul_adds_per_elem = conv_activation_c * filter_h * filter_w * 2;  // 1 multiply and 1 add per element
    int64_t num_mul_adds = num_mul_adds_per_elem * output_height * output_width * args.output_channels * batch_size;

    int ideal_dev_clock_cycles = std::ceil(
        ((float)num_mul_adds / (float)(num_cores * tensix_mul_adds_per_cycle_lofi)) *
        (float)tt::tt_metal::operation::OpPerformanceModel::fidelity_multiplier(
            get_math_fidelity(args.compute_kernel_config)));

    Tensors input_tensors = {tensor_args.a, tensor_args.b};
    if (tensor_args.bias.has_value()) {
        input_tensors.push_back(tensor_args.bias.value());
    }
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        input_tensors, output_tensor, ideal_dev_clock_cycles);

#if 0
    log_info(tt::LogOp, "Conv2d PerfModel:");
    log_info(tt::LogOp, "\t Batch: {}", batch_size);
    log_info(tt::LogOp, "\t In (H, W, C): ({}, {}, {})", conv_activation_h, conv_activation_w, conv_activation_c);
    log_info(tt::LogOp, "\t Filter (H, W): ({}, {})", filter_h, filter_w);
    log_info(tt::LogOp, "\t Filter Stride (H, W): ({}, {})", stride_h, stride_w);
    log_info(tt::LogOp, "\t Pad (H, W): ({}, {})", pad_h, pad_w);
    log_info(tt::LogOp, "\t Out (H, W, C): ({}, {}, {})", output_height, output_width, this->output_channels);
    log_info(tt::LogOp, "\t ideal_dev_clock_cycles: {}", ideal_dev_clock_cycles);
#endif

    return result;
}

}  // namespace ttnn::operations::conv::conv2d

namespace ttnn::prim {

ttnn::operations::conv::conv2d::Conv2dDeviceOperation::tensor_return_value_t conv2d(
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    const ttnn::operations::sliding_window::SlidingWindowConfig& sliding_window_config,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation,
    const ttnn::operations::conv::conv2d::Conv2dParallelizationConfig& parallelization_config,
    const ttnn::operations::conv::conv2d::Conv2dBlockConfig& block_config,
    const tt::tt_metal::MemoryConfig& memory_config,
    tt::tt_metal::DataType dtype,
    std::array<std::uint32_t, 4> input_tensor_shape,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    bool enable_act_double_buffer,
    bool enable_weights_double_buffer,
    bool full_inner_dim,
    bool enable_activation_reuse,
    bool config_tensors_in_dram,
    std::optional<bool> force_split_reader) {
    using OperationType = ttnn::operations::conv::conv2d::Conv2dDeviceOperation;

    TT_FATAL(b.layout() == Layout::TILE, "Weights should be in TILE layout.");

    auto operation_attributes = OperationType::operation_attributes_t{
        .sliding_window_config = sliding_window_config,
        .output_channels = output_channels,
        .groups = groups,
        .untilize_out = untilize_out,
        .has_bias = bias.has_value(),
        .activation = activation,
        .parallelization_config = parallelization_config,
        .block_config = block_config,
        .memory_config = memory_config,
        .dtype = dtype,
        .input_tensor_shape = input_tensor_shape,
        .compute_kernel_config = compute_kernel_config,
        .enable_act_double_buffer = enable_act_double_buffer,
        .enable_weights_double_buffer = enable_weights_double_buffer,
        .full_inner_dim = full_inner_dim,
        .enable_activation_reuse = enable_activation_reuse,
        .config_tensors_in_dram = config_tensors_in_dram,
        .force_split_reader = force_split_reader,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .a = a,
        .b = b,
        .bias = bias,
    };

    auto* device = a.device();

    operation_attributes.pre_op_l1_allocation_size_bytes =
        device->allocator()->get_statistics(tt::tt_metal::BufferType::L1).total_allocated_bytes;

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
