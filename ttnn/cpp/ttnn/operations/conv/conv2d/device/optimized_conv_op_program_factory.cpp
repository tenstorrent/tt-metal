// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include "optimized_conv_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/constants.hpp"

#include "tt_metal/tt_stl/reflection.hpp"

#include "tt_metal/common/work_split.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/sharding_utilities.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"

#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
using namespace tt::constants;
namespace optimized_conv_op_utils {
using namespace tt;
using namespace tt::tt_metal;

pair<uint32_t, uint32_t> compute_opt_conv_output_face_shape(uint32_t conv_activation_h, uint32_t conv_activation_w, uint32_t filter_h, uint32_t filter_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t padding_for_32B_alignment) {
    uint32_t conv_output_h = ((conv_activation_h - filter_h + (2 * pad_h)) / stride_h) + 1;
    uint32_t conv_output_w = ((conv_activation_w - filter_w + (2 * pad_w) - padding_for_32B_alignment) / stride_w) + 1;
    return {conv_output_h, conv_output_w};
}
pair<vector<uint32_t>, vector<uint32_t>> compute_opt_conv_activation_as_mm_shape(const Shape& conv_activation_shape, vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t padding_for_32B_alignment) {
    uint32_t filter_h = (uint32_t) conv_params[0];
    uint32_t filter_w = (uint32_t) conv_params[1];
    uint32_t stride_h = (uint32_t) conv_params[2];
    uint32_t stride_w = (uint32_t) conv_params[3];
    uint32_t pad_h = (uint32_t) conv_params[4];
    uint32_t pad_w = (uint32_t) conv_params[5];
    auto [conv_output_h, conv_output_w] = compute_opt_conv_output_face_shape(conv_activation_shape[1], conv_activation_shape[2], filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, padding_for_32B_alignment);
    uint32_t batch_size = conv_activation_shape[0];
    // pad height
    uint32_t num_rows = (uint32_t) batch_size * conv_output_h * conv_output_w;
    uint32_t act_block_h_datums = act_block_h_ntiles * TILE_HEIGHT;
    uint32_t num_rows_padded = (uint32_t) (std::ceil((double) num_rows / (double) act_block_h_datums ) * act_block_h_datums);
    uint32_t num_cols = conv_activation_shape[3] * filter_h * filter_w;
    uint32_t num_cols_padded = round_up(conv_activation_shape[3] * filter_w, TILE_WIDTH) * filter_h;
    return {{1, num_rows_padded, num_cols_padded}, {1, num_rows, num_cols}};
}

pair<vector<uint32_t>, vector<uint32_t>> compute_opt_conv_activation_as_mm_shape(const Shape& conv_activation_shape, ttnn::operations::sliding_window::SlidingWindowConfig sliding_window_config, uint32_t act_block_h_ntiles, uint32_t padding_for_32B_alignment) {

    uint32_t filter_h = (uint32_t)sliding_window_config.window_hw.first;  // filter_h
    uint32_t filter_w = (uint32_t)sliding_window_config.window_hw.second;  // filter_W
    auto output_shape = sliding_window_config.get_output_shape();
    uint32_t batch_size = output_shape[0];
    uint32_t conv_output_h = output_shape[1];
    uint32_t conv_output_w = output_shape[2];

    // pad height
    uint32_t num_rows = (uint32_t) batch_size * conv_output_h * conv_output_w;
    uint32_t act_block_h_datums = act_block_h_ntiles * TILE_HEIGHT;
    uint32_t num_rows_padded = (uint32_t) (std::ceil((double) num_rows / (double) act_block_h_datums ) * act_block_h_datums);
    uint32_t num_cols = conv_activation_shape[3] * filter_h * filter_w;
    uint32_t num_cols_padded = round_up(conv_activation_shape[3] * filter_w, TILE_WIDTH) * filter_h;
    return {{1, num_rows_padded, num_cols_padded}, {1, num_rows, num_cols}};
}

} // optimized_conv_op_utils

namespace ttnn::operations::conv {
namespace conv2d {

Tensor optimized_conv_new(const Tensor& a, const Tensor &b, std::optional<const Tensor> bias,
    sliding_window::SlidingWindowConfig sliding_window_config,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out, bool fuse_relu, MathFidelity math_fidelity,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config, uint32_t extra_padding_for_32B_alignment,
    MemoryConfig memory_config,
    DataType dtype,
    std::array<std::uint32_t, 4> input_tensor_shape,
    bool use_shallow_conv_variant,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool enable_act_double_buffer,
    bool enable_split_reader,
    bool enable_subblock_padding
) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({a, b}))};
    operation::launch_op(
        [sliding_window_config, output_channels, groups, untilize_out, fuse_relu, math_fidelity, parallelization_config, block_config, extra_padding_for_32B_alignment, memory_config, dtype, input_tensor_shape, use_shallow_conv_variant, compute_kernel_config, enable_act_double_buffer, enable_split_reader, enable_subblock_padding]
            (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                using ttnn::operations::experimental::auto_format::FormatParams;
                auto& a = input_tensors.at(0);
                auto& b = input_tensors.at(1);
                auto& bias = optional_input_tensors.at(0);
                //TT_ASSERT(!untilize_out, "Optimized conv only supports tiled out");
                TT_ASSERT(b.get_layout() == Layout::TILE); // Weights should already be formatted
                const auto& ashape = Shape(input_tensor_shape);
                auto padded_a_shape = Shape(std::array<uint32_t,4>{ashape[0], ashape[1], ashape[2], tt::round_up(ashape[3], 16)});
                FormatParams input_a_format_params = {.pad_shape=padded_a_shape.value, .pad_value=0.0, .target_layout=Layout::ROW_MAJOR};
                FormatParams input_b_format_params = {.pad_shape=b.get_legacy_shape(), .pad_value=0.0, .target_layout=Layout::TILE};
                FormatParams input_bias_format_params = {};
                if (bias.has_value()) {
                    input_bias_format_params = {.pad_shape=bias.value().get_legacy_shape(), .pad_value=0, .target_layout=Layout::TILE};
                }
                auto output_layout = untilize_out ? Layout::ROW_MAJOR : Layout::TILE;
                auto arch = is_tensor_on_device_or_multidevice(a) ? a.device()->arch() : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
                bool fp32_accum = a.device()->arch() == tt::ARCH::WORMHOLE_B0;  // && compute_kernel_config.has_value()) ? compute_kernel_config.value().fp32_dest_acc_en : false;
                auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::LoFi, true, fp32_accum, false);
                return operation::run_without_autoformat(
                    OptimizedConvNew(sliding_window_config, output_channels, groups, untilize_out, bias.has_value(), fuse_relu, math_fidelity, parallelization_config, block_config, extra_padding_for_32B_alignment, memory_config, dtype, input_tensor_shape, use_shallow_conv_variant, kernel_config_val, enable_act_double_buffer, enable_split_reader, enable_subblock_padding
                    ),
                    input_tensors,
                    optional_input_tensors);
            }, {a, b}, output_tensors, {bias});
    return output_tensors.at(0);

}

void OptimizedConvNew::validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    // TODO: ...
    TT_FATAL(!input_tensor_b.memory_config().is_sharded(), "Error");
    if (this->untilize_out) {
        TT_FATAL((this->dtype == DataType::BFLOAT16) || (this->dtype == DataType::FLOAT32), "Error");
    }
    if (this->memory_config.is_sharded()) {
        uint32_t out_block_h_ntiles = parallelization_config.per_core_out_matrix_height_ntiles;
        auto [act_matrix_shape, act_matrix_shape_unpadded] = optimized_conv_op_utils::compute_opt_conv_activation_as_mm_shape(input_tensor_a.get_legacy_shape(), sliding_window_config, out_block_h_ntiles, extra_padding_for_32B_alignment);
        uint32_t out_width_ntiles = this->compute_output_shapes(input_tensors).at(0)[-1] / TILE_WIDTH;
        if(this->memory_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            TT_FATAL(this->parallelization_config.per_core_out_matrix_width_ntiles == out_width_ntiles, "Error");
            TT_FATAL(this->block_config.out_subblock_w_ntiles == out_width_ntiles || this->block_config.out_subblock_h_ntiles == 1, "Error");
        } else if (this->memory_config.memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            // For block sharded, out_width per core is shard width, and this is split along row
            // TODO: We should clean this up and relax constraints on out_subblock h and w
            if (this->memory_config.shard_spec.value().orientation == ShardOrientation::COL_MAJOR) {
                out_width_ntiles /= this->parallelization_config.grid_size.y;
            } else {
                out_width_ntiles /= this->parallelization_config.grid_size.x;
            }
            TT_FATAL(this->block_config.out_subblock_w_ntiles == out_width_ntiles || this->block_config.out_subblock_h_ntiles == 1, "Error");
        }
    }
}

std::vector<tt::tt_metal::Shape> OptimizedConvNew::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a_shape = this->input_tensor_shape;
    uint32_t batch_size = input_tensor_a_shape[0];
    uint32_t conv_activation_h = input_tensor_a_shape[1];
    uint32_t conv_activation_w = input_tensor_a_shape[2];
    // TODO: clean up here
    uint32_t filter_h = (uint32_t)sliding_window_config.window_hw.first;  // filter_h
    uint32_t filter_w = (uint32_t)sliding_window_config.window_hw.second;  // filter_W
    uint32_t stride_h = (uint32_t)sliding_window_config.stride_hw.first;
    uint32_t stride_w = (uint32_t)sliding_window_config.stride_hw.second;
    uint32_t pad_h = (uint32_t)sliding_window_config.pad_hw.first;
    uint32_t pad_w = (uint32_t)sliding_window_config.pad_hw.second;
    auto [conv_output_h, conv_output_w] = optimized_conv_op_utils::compute_opt_conv_output_face_shape(conv_activation_h, conv_activation_w, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, extra_padding_for_32B_alignment);

    // Tiled output shape is padded shape. Padded to tile shape.
    auto shape_w = batch_size * conv_output_h * conv_output_w;
    auto shape_c = output_channels;
    auto padded_shape_w =
        parallelization_config.num_cores_nhw * parallelization_config.per_core_out_matrix_height_ntiles * TILE_HEIGHT;
    auto padded_shape_c = tt::round_up(this->output_channels, TILE_WIDTH);
    auto output_padding = Padding(
        {{0, 0}, {0, 0}, {0, (padded_shape_w - shape_w)}, {0, (padded_shape_c - shape_c)}}, Padding::PadValue::Zero);
    auto output_tensor_shape = Shape(tt::tt_metal::Shape({1, 1, padded_shape_w, padded_shape_c}, output_padding));
    return {output_tensor_shape.value};
}

std::vector<Tensor> OptimizedConvNew::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& weight_tensor = input_tensors.at(1);
    auto output_layout = this->untilize_out ? Layout::ROW_MAJOR : Layout::TILE;
    if (this->memory_config.is_sharded()) {
        auto output_shape = this->compute_output_shapes(input_tensors).at(0);
        if (this->memory_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            uint32_t total_height_tiles = tt::tt_metal::compute_volume(output_shape) / output_shape[-1] / TILE_HEIGHT;
            uint32_t num_cores = total_height_tiles / this->parallelization_config.per_core_out_matrix_height_ntiles;
            CoreRangeSet shard_grid = tt::tt_metal::num_cores_to_corerange_set(num_cores, this->parallelization_config.grid_size, true);

            std::array<uint32_t, 2> shard_shape = {this->parallelization_config.per_core_out_matrix_height_ntiles * TILE_HEIGHT, output_shape[-1]};
            auto shard_spec = ShardSpec{shard_grid, shard_shape, ShardOrientation::ROW_MAJOR};
            auto mem_config = this->memory_config;
            mem_config.shard_spec = shard_spec;
            return {create_device_tensor(output_shape, this->dtype, output_layout, input_tensor.device(), mem_config)};
        } else if(this->memory_config.memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            uint32_t total_height_tiles = tt::tt_metal::compute_volume(output_shape) / output_shape[-1] / TILE_HEIGHT;
            std::array<uint32_t, 2> shard_shape = {this->parallelization_config.per_core_out_matrix_height_ntiles * TILE_HEIGHT, this->parallelization_config.per_core_out_matrix_width_ntiles * TILE_WIDTH};
            auto shard_grid = input_tensor.memory_config().shard_spec.value().grid;
            auto shard_spec = ShardSpec{shard_grid, shard_shape, this->memory_config.shard_spec.value().orientation};
            auto mem_config = this->memory_config;
            mem_config.shard_spec = shard_spec;
            return{create_device_tensor(output_shape, this->dtype, output_layout, input_tensor.device(), mem_config)};

        } else if (this->memory_config.memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            auto [act_matrix_shape, act_matrix_shape_unpadded] = optimized_conv_op_utils::compute_opt_conv_activation_as_mm_shape(this->input_tensor_shape, sliding_window_config, this->parallelization_config.per_core_out_matrix_height_ntiles, extra_padding_for_32B_alignment);
            uint32_t act_matrix_height = (uint32_t) act_matrix_shape[1];
            uint32_t act_matrix_height_ntiles = act_matrix_height / TILE_HEIGHT;
            uint32_t total_active_num_cores_per_weight_slice = act_matrix_height_ntiles / this->parallelization_config.per_core_out_matrix_height_ntiles;
            uint32_t weight_matrix_width = weight_tensor.get_legacy_shape()[-1];
            uint32_t weight_matrix_width_ntiles = weight_matrix_width / TILE_WIDTH;
            uint32_t num_weight_slices_width = weight_matrix_width_ntiles / this->parallelization_config.per_core_out_matrix_width_ntiles ;
            uint32_t total_active_num_cores = total_active_num_cores_per_weight_slice * num_weight_slices_width;
            log_debug(tt::LogOp, "Total active num cores: {}", total_active_num_cores);
            log_debug(tt::LogOp, "Parallelization config grid size: {}", this->parallelization_config.grid_size.str());
            uint32_t num_cores_x = this->parallelization_config.grid_size.x;
            uint32_t num_cores_y = this->parallelization_config.grid_size.y;
            CoreRangeSet shard_grid = CoreRangeSet({{{0, 0}, {num_cores_x - 1, num_cores_y - 1}}});
            log_debug(tt::LogOp, "Calculated shard_grid: {}", shard_grid.str());
            std::array<uint32_t, 2> shard_shape = {this->parallelization_config.per_core_out_matrix_height_ntiles * TILE_HEIGHT, this->parallelization_config.per_core_out_matrix_width_ntiles * TILE_WIDTH};
            auto shard_spec = ShardSpec{shard_grid, shard_shape, this->memory_config.shard_spec.value().orientation};
            auto mem_config = this->memory_config;
            mem_config.shard_spec = shard_spec;
            return {create_device_tensor(output_shape, this->dtype, output_layout, input_tensor.device(), mem_config)};
        } else {
            TT_THROW("Unsupported shard scheme");
        }

    }
    return operation::generic_create_output_tensors(*this, input_tensors, this->dtype, output_layout, this->memory_config);
}

operation::ProgramWithCallbacks OptimizedConvNew::create_program(const std::vector<Tensor>& input_tensors,
                                                     const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                                     std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& input_tensor_bias = optional_input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    TT_ASSERT(input_tensor_a.memory_config().is_sharded()); // TODO: move this check to validate_input_tensors
    return multi_core_optimized_conv_sharded_v2_new(
        input_tensor_a, input_tensor_b, input_tensor_bias,
        sliding_window_config,
        output_channels,
        groups,
        untilize_out, fuse_relu, math_fidelity,
        parallelization_config,
        block_config, extra_padding_for_32B_alignment,
        dtype,
        input_tensor_shape,
        use_shallow_conv_variant,
        compute_kernel_config,
        output_tensor,
        enable_act_double_buffer,
        enable_split_reader,
        enable_subblock_padding);
}

operation::OpPerformanceModel OptimizedConvNew::create_op_performance_model(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a_shape = this->input_tensor_shape;
    uint32_t batch_size = input_tensor_a_shape[0];
    uint32_t conv_activation_h = input_tensor_a_shape[1];
    uint32_t conv_activation_w = input_tensor_a_shape[2];
    uint32_t conv_activation_c = input_tensor_a_shape[3];
    uint32_t filter_h = (uint32_t)sliding_window_config.window_hw.first;  // filter_h
    uint32_t filter_w = (uint32_t)sliding_window_config.window_hw.second;  // filter_W
    uint32_t stride_h = (uint32_t)sliding_window_config.stride_hw.first;
    uint32_t stride_w = (uint32_t)sliding_window_config.stride_hw.second;
    uint32_t pad_h = (uint32_t)sliding_window_config.pad_hw.first;
    uint32_t pad_w = (uint32_t)sliding_window_config.pad_hw.second;

    const auto& t = output_tensors.at(0);
    if(t.storage_type() != StorageType::DEVICE) {
        tt::log_warning(tt::LogOp, "Output tensor not on DEVICE?!");
    }

    auto arch = t.storage_type() == StorageType::DEVICE ? t.device()->arch() : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    const int num_cores = (arch == tt::ARCH::WORMHOLE_B0) ? 8 * 8 : 9 * 12;
    const int tensix_mul_adds_per_cycle_lofi = (arch == tt::ARCH::WORMHOLE_B0) ? 4096 : 2048;

    // Calculate output dimensions: relevant for window/stride based OPs (conv, maxpool, downsample)
    int output_height = std::floor((conv_activation_h - filter_h + 2 * pad_h) / stride_h + 1);
    int output_width = std::floor((conv_activation_w - filter_w + 2 * pad_w) / stride_w + 1);

    // Calculate number of mul/add operations
    // TODO: add bias modeling
    int64_t num_mul_adds_per_elem = conv_activation_c * filter_h * filter_w * 2; // 1 multiply and 1 add per element
    int64_t num_mul_adds = num_mul_adds_per_elem * output_height * output_width * this->output_channels * batch_size;

    int ideal_dev_clock_cycles = std::ceil(((float)num_mul_adds / (float)(num_cores * tensix_mul_adds_per_cycle_lofi)) * (float)operation::OpPerformanceModel::fidelity_multiplier(this->math_fidelity));

    operation::OpPerformanceModel result(input_tensors, output_tensors, ideal_dev_clock_cycles);

#if 0
    tt::log_info(tt::LogOp, "OptimizedConv PerfModel:");
    tt::log_info(tt::LogOp, "\t Batch: {}", batch_size);
    tt::log_info(tt::LogOp, "\t In (H, W, C): ({}, {}, {})", conv_activation_h, conv_activation_w, conv_activation_c);
    tt::log_info(tt::LogOp, "\t Filter (H, W): ({}, {})", filter_h, filter_w);
    tt::log_info(tt::LogOp, "\t Filter Stride (H, W): ({}, {})", stride_h, stride_w);
    tt::log_info(tt::LogOp, "\t Pad (H, W): ({}, {})", pad_h, pad_w);
    tt::log_info(tt::LogOp, "\t Out (H, W, C): ({}, {}, {})", output_height, output_width, this->output_channels);
    tt::log_info(tt::LogOp, "\t ideal_dev_clock_cycles: {}", ideal_dev_clock_cycles);
#endif

    return result;
}
}  // namespace tt_metal

}  // namespace tt
