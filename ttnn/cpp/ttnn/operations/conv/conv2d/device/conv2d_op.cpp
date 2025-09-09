// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include "conv2d_op.hpp"
#include <tt-metalium/math.hpp>

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/constants.hpp>

#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/operations/cb_utils.hpp"

#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/shape/shape.hpp"

namespace ttnn::operations::conv {
namespace conv2d {

Tensor optimized_conv_new(
    const Tensor& a,
    const Tensor& b,
    std::optional<const Tensor> bias,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out,
    const std::string& activation,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config,
    const MemoryConfig& memory_config,
    DataType dtype,
    std::array<std::uint32_t, 4> input_tensor_shape,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool enable_act_double_buffer,
    bool enable_weights_double_buffer,
    bool full_inner_dim,
    bool enable_split_reader) {
    TT_FATAL(b.layout() == Layout::TILE,
             "Weights should be in TILE layout.");  // Weights should already be formatted
    const auto& ashape = input_tensor_shape;
    auto padded_a_shape = ttnn::Shape({ashape[0], ashape[1], ashape[2], tt::round_up(ashape[3], 16)});
    experimental::auto_format::FormatParams input_a_format_params = {
        .pad_shape = padded_a_shape, .pad_value = 0.0, .target_layout = Layout::ROW_MAJOR};
    experimental::auto_format::FormatParams input_b_format_params = {
        .pad_shape = b.padded_shape(), .pad_value = 0.0, .target_layout = Layout::TILE};
    experimental::auto_format::FormatParams input_bias_format_params = {};
    if (bias.has_value()) {
        input_bias_format_params = {
            .pad_shape = bias.value().padded_shape(), .pad_value = 0, .target_layout = Layout::TILE};
    }
    auto optimized_conv_op = OptimizedConvNew(
        sliding_window_config,
        output_channels,
        groups,
        untilize_out,
        bias.has_value(),
        activation,
        parallelization_config,
        block_config,
        memory_config,
        dtype,
        input_tensor_shape,
        compute_kernel_config,
        enable_act_double_buffer,
        enable_weights_double_buffer,
        full_inner_dim,
        enable_split_reader);
    IDevice* device = a.device();

    optimized_conv_op.pre_op_l1_allocation_size_bytes =
        device->allocator()->get_statistics(tt::tt_metal::BufferType::L1).total_allocated_bytes;
    return tt::tt_metal::operation::run_without_autoformat(optimized_conv_op, {a, b}, {bias}).at(0);
}

void OptimizedConvNew::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    TT_FATAL(input_tensor_a.memory_config().is_sharded(), "Activation tensor should be sharded.");
    TT_FATAL(!input_tensor_b.memory_config().is_sharded(), "Weights tensor should not be sharded.");
    if (this->untilize_out) {
        TT_FATAL((this->dtype == DataType::BFLOAT16) || (this->dtype == DataType::FLOAT32), "Error");
    }
    if (this->memory_config.is_sharded()) {
        uint32_t per_core_out_matrix_width_ntiles = parallelization_config.per_core_out_matrix_width_ntile;
        uint32_t out_width_ntiles =
            this->compute_output_specs(input_tensors).at(0).padded_shape()[-1] / tt::constants::TILE_WIDTH;
        if (this->memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
            TT_FATAL(per_core_out_matrix_width_ntiles == out_width_ntiles, "Error");
            TT_FATAL(
                this->block_config.out_subblock_w_ntiles == out_width_ntiles ||
                    this->block_config.out_subblock_h_ntiles == 1,
                "Error");
        } else if (this->memory_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            // For block sharded, out_width per core is shard width, and this is split along row
            // TODO: We should clean this up and relax constraints on out_subblock h and w
            if (this->memory_config.shard_spec().value().orientation == ShardOrientation::COL_MAJOR) {
                out_width_ntiles = tt::div_up(out_width_ntiles, this->parallelization_config.grid_size.y);
            } else {
                out_width_ntiles = tt::div_up(out_width_ntiles, this->parallelization_config.grid_size.x);
            }
        }
        TT_FATAL(
            this->block_config.out_subblock_w_ntiles == per_core_out_matrix_width_ntiles ||
                this->block_config.out_subblock_h_ntiles == 1,
            "Error");
    }
}

std::vector<TensorSpec> OptimizedConvNew::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a_shape = this->input_tensor_shape;
    uint32_t batch_size = input_tensor_a_shape[0];

    auto sliding_window_output_shape = sliding_window_config.get_output_shape();
    uint32_t conv_output_h = sliding_window_output_shape[1];
    uint32_t conv_output_w = sliding_window_output_shape[2];

    // Tiled output shape is padded shape. Padded to tile shape.
    auto shape_w = batch_size * conv_output_h * conv_output_w;
    auto shape_c = output_channels;
    auto padded_shape_w = parallelization_config.num_cores_nhw *
                          parallelization_config.per_core_out_matrix_height_ntile * tt::constants::TILE_HEIGHT;
    auto padded_shape_c = tt::round_up(this->output_channels, tt::constants::TILE_WIDTH);
    ttnn::Shape output_shape({1, 1, shape_w, shape_c});
    ttnn::Shape padded_output_shape({1, 1, padded_shape_w, padded_shape_c});

    auto output_layout = this->untilize_out ? Layout::ROW_MAJOR : Layout::TILE;
    if (this->memory_config.is_sharded()) {
        std::array<uint32_t, 2> shard_shape = {
            this->parallelization_config.per_core_out_matrix_height_ntile * tt::constants::TILE_HEIGHT,
            this->parallelization_config.per_core_out_matrix_width_ntile * tt::constants::TILE_WIDTH};
        auto shard_grid = this->memory_config.shard_spec().value().grid;
        auto shard_spec =
            tt::tt_metal::ShardSpec{shard_grid, shard_shape, this->memory_config.shard_spec().value().orientation};
        auto mem_config = this->memory_config.with_shard_spec(shard_spec);
        return {TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                dtype,
                tt::tt_metal::PageConfig(output_layout),
                mem_config,
                tt::tt_metal::Alignment(
                    {tt::constants::TILE_HEIGHT,
                     tt::constants::TILE_WIDTH})  // Conv2D always outputs in tile multiples, even if output layout is
                                                  // Row Major.
                ))};
    }
    return {TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout::fromPaddedShape(
            dtype, tt::tt_metal::PageConfig(output_layout), memory_config, output_shape, padded_output_shape))};
}

tt::tt_metal::operation::ProgramWithCallbacks OptimizedConvNew::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& input_tensor_bias = optional_input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    tt::tt_metal::IDevice* device = input_tensor_a.device();

    const bool has_bias = input_tensor_bias.has_value();

    const auto& weights_shape = input_tensor_b.padded_shape();

    std::optional<unary::UnaryWithParam> fused_activation = std::nullopt;

    if (!activation.empty()) {
        fused_activation = unary::utils::string_to_unary_with_param(activation);
    }

    // Factory selection logic - choose the appropriate implementation based on memory layout
    tt::tt_metal::operation::ProgramWithCallbacks program_with_cbs;

    if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        // Use width sharded implementation
        tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

        ttnn::operations::sliding_window::ParallelConfig parallel_config{
            .grid = input_tensor_a.shard_spec().value().grid,
            .shard_scheme = input_tensor_a.memory_config().memory_layout(),
            .shard_orientation = input_tensor_a.shard_spec().value().orientation};

        std::vector<uint32_t> op_trace_metadata =
            ttnn::operations::sliding_window::generate_op_trace_metadata(sliding_window_config);
        std::vector<sliding_window::ShardBoundary> shard_boundaries =
            ttnn::operations::sliding_window::generate_shard_boundaries(sliding_window_config, op_trace_metadata);

        program_with_cbs = multi_core_optimized_conv_width_sharded_v2_impl(
            program,
            input_tensor_a,
            input_tensor_b,
            ttnn::Shape(input_tensor_shape),
            input_tensor_bias,
            sliding_window_config,
            parallel_config,
            op_trace_metadata,
            shard_boundaries,
            output_channels,
            groups,
            untilize_out,
            has_bias,
            fused_activation,
            parallelization_config,
            block_config,
            output_tensor,
            compute_kernel_config,
            enable_act_double_buffer,
            enable_weights_double_buffer);
    } else {
        // Use regular sharded implementation
        tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

        ttnn::operations::sliding_window::ParallelConfig parallel_config{
            .grid = input_tensor_a.shard_spec().value().grid,
            .shard_scheme = input_tensor_a.memory_config().memory_layout(),
            .shard_orientation = input_tensor_a.shard_spec().value().orientation};

        std::vector<uint32_t> op_trace_metadata =
            ttnn::operations::sliding_window::generate_op_trace_metadata(sliding_window_config);
        std::vector<sliding_window::ShardBoundary> shard_boundaries =
            ttnn::operations::sliding_window::generate_shard_boundaries(sliding_window_config, op_trace_metadata);

        program_with_cbs = multi_core_optimized_conv_sharded_v2_impl(
            program,
            input_tensor_a,
            input_tensor_b,
            ttnn::Shape(input_tensor_shape),
            input_tensor_bias,
            sliding_window_config,
            parallel_config,
            op_trace_metadata,
            shard_boundaries,
            output_channels,
            groups,
            untilize_out,
            has_bias,
            fused_activation,
            parallelization_config,
            block_config,
            input_tensor_a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR,
            output_tensor,
            compute_kernel_config,
            enable_act_double_buffer,
            enable_weights_double_buffer,
            enable_split_reader,
            full_inner_dim);
    }

    const uint32_t post_op_l1_allocation_size =
        device->allocator()->get_statistics(tt::tt_metal::BufferType::L1).total_allocated_bytes;

    auto actual_cb_size = calculate_total_cb_size(program_with_cbs.program);

    auto kernel_dims =
        std::array<uint32_t, 2>({sliding_window_config.window_hw.first, sliding_window_config.window_hw.second});

    const SkipMcast skip_mcast = conv_skip_mcast(parallelization_config, memory_config.memory_layout());
    conv_op_l1_usage l1_usage = calculate_L1_usage(
        compute_kernel_config,
        block_config,
        parallelization_config,
        weights_shape,
        std::array<uint32_t, 2>({sliding_window_config.window_hw.first, sliding_window_config.window_hw.second}),
        Conv2dConfig{
            .weights_dtype = input_tensor_b.dtype(),
            .shard_layout = this->memory_config.memory_layout(),
            .output_layout = (untilize_out ? Layout::ROW_MAJOR : Layout::TILE),
            .enable_act_double_buffer = enable_act_double_buffer,
            .enable_weights_double_buffer = enable_weights_double_buffer,
            .enable_split_reader = enable_split_reader},
        input_tensor_a.dtype(),
        this->dtype,
        has_bias,
        is_1d_deptwise_conv(
            groups,
            input_tensor_shape[3],
            output_channels,
            kernel_dims[1],
            sliding_window_config.get_output_shape()[2],
            has_bias),
        skip_mcast.skip_activation_mcast);

    TT_FATAL(
        actual_cb_size == l1_usage.CB_allocation_size,
        "Calculated CB size {} does not match with the actual CB size {}",
        l1_usage.CB_allocation_size,
        actual_cb_size);

    // For now assume that if post_op_l1_allocation_size == 0 op is being run
    // in graph capture NO_DISPATCH mode.
    // ToDo: Device should offer an API to inform the op if it is running in NO_DISPATCH mode.
    bool is_graph_capture_no_dispathch_mode = post_op_l1_allocation_size == 0;
    TT_FATAL(
        post_op_l1_allocation_size == (this->pre_op_l1_allocation_size_bytes + l1_usage.tensor_allocation_size) ||
            is_graph_capture_no_dispathch_mode,
        "Mismatch!! L1 Allocation Pre Op =  {}, Post Op = {} Calculated Size = {}",
        this->pre_op_l1_allocation_size_bytes,
        post_op_l1_allocation_size,
        l1_usage.tensor_allocation_size);
    return program_with_cbs;
}

tt::tt_metal::operation::OpPerformanceModel OptimizedConvNew::create_op_performance_model(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a_shape = this->input_tensor_shape;
    uint32_t batch_size = input_tensor_a_shape[0];
    uint32_t conv_activation_h = input_tensor_a_shape[1];
    uint32_t conv_activation_w = input_tensor_a_shape[2];
    uint32_t conv_activation_c = input_tensor_a_shape[3];
    uint32_t filter_h = (uint32_t)sliding_window_config.window_hw.first;   // filter_h
    uint32_t filter_w = (uint32_t)sliding_window_config.window_hw.second;  // filter_W
    uint32_t stride_h = (uint32_t)sliding_window_config.stride_hw.first;
    uint32_t stride_w = (uint32_t)sliding_window_config.stride_hw.second;
    uint32_t dilation_h = (uint32_t)sliding_window_config.dilation_hw.first;
    uint32_t dilation_w = (uint32_t)sliding_window_config.dilation_hw.second;

    const auto& t = output_tensors.at(0);
    if (t.storage_type() != StorageType::DEVICE) {
        log_warning(tt::LogOp, "Output tensor not on DEVICE?!");
    }

    auto arch = t.storage_type() == StorageType::DEVICE
                    ? t.device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    const int num_cores = (arch == tt::ARCH::WORMHOLE_B0) ? 8 * 8 : 9 * 12;
    const int tensix_mul_adds_per_cycle_lofi = (arch == tt::ARCH::WORMHOLE_B0) ? 4096 : 2048;

    // Calculate output dimensions: relevant for window/stride based OPs (conv, maxpool, downsample)
    auto [output_height, output_width] = calculate_output_image_size(
        {conv_activation_h, conv_activation_w},
        {filter_h, filter_w},
        {stride_h, stride_w},
        sliding_window_config.padding,
        {dilation_h, dilation_w});

    // Calculate number of mul/add operations
    // TODO: add bias modeling
    int64_t num_mul_adds_per_elem = conv_activation_c * filter_h * filter_w * 2;  // 1 multiply and 1 add per element
    int64_t num_mul_adds = num_mul_adds_per_elem * output_height * output_width * this->output_channels * batch_size;

    int ideal_dev_clock_cycles = std::ceil(
        ((float)num_mul_adds / (float)(num_cores * tensix_mul_adds_per_cycle_lofi)) *
        (float)tt::tt_metal::operation::OpPerformanceModel::fidelity_multiplier(
            get_math_fidelity(this->compute_kernel_config)));

    tt::tt_metal::operation::OpPerformanceModel result(input_tensors, output_tensors, ideal_dev_clock_cycles);

#if 0
    log_info(tt::LogOp, "OptimizedConv PerfModel:");
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
}  // namespace conv2d

}  // namespace ttnn::operations::conv
