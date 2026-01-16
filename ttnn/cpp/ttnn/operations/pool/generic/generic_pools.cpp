// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_pools.hpp"

#include "tt-metalium/constants.hpp"
#include "ttnn/operations/pool/generic/device/pool_op.hpp"
#include <cmath>
#include <optional>
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::operations::pool {

// Generic invoke function for both max and avg pool operations. Most of the arguments are shared excpet for the
// dilation which is set to (1,1) for avg pool and count_include_pad and divisor_override which have no effect on
// maxpool.

static std::tuple<MemoryConfig, uint32_t, sliding_window::ParallelConfig> get_pool_input_memory_config(
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    const std::optional<const TensorMemoryLayout> applied_shard_scheme,
    uint32_t batch_size,
    uint32_t channels,
    const ttnn::Shape& input_shape,
    const ttnn::Shape& output_shape,
    const tt::tt_metal::CoreCoord& core_grid,
    const DataType& input_dtype,
    const Layout& input_layout,
    const DataType& output_dtype,
    const Layout& output_layout,
    Pool2DType pool_type,
    bool count_include_pad,
    std::optional<int32_t> divisor_override,
    bool return_indices,
    bool config_tensor_in_dram) {
    bool is_out_tiled = output_layout == ttnn::TILE_LAYOUT;
    bool is_in_tiled = input_layout == ttnn::TILE_LAYOUT;
    sliding_window::ParallelConfig parallel_config;

    uint32_t smallest_RM_elem_size =
        tt::datum_size(tt::tt_metal::datatype_to_dataformat_converter(DataType::BFLOAT16));  // Size of BFloat16
    uint32_t input_channels_alignment =
        is_in_tiled ? tt::constants::TILE_WIDTH : (tt::tt_metal::hal::get_l1_alignment() / smallest_RM_elem_size);
    TensorMemoryLayout shard_layout = TensorMemoryLayout::HEIGHT_SHARDED;  // default to height sharding
    if (applied_shard_scheme.has_value()) {
        TT_FATAL(
            (applied_shard_scheme.value() == TensorMemoryLayout::HEIGHT_SHARDED) ||
                (applied_shard_scheme.value() == TensorMemoryLayout::WIDTH_SHARDED) ||
                (applied_shard_scheme.value() == TensorMemoryLayout::BLOCK_SHARDED),
            "Only height, width, or block sharding strategies are supported.");
        shard_layout = applied_shard_scheme.value();
        parallel_config = conv::determine_parallel_config(
            shard_layout,
            batch_size,
            channels,
            output_shape[1],
            output_shape[2],
            channels,
            input_channels_alignment,
            core_grid,
            ShardOrientation::ROW_MAJOR,
            false,
            is_out_tiled,
            is_in_tiled || is_out_tiled,  // if input/output is tiled we need to choose num_cores_c to make the
                                          // shard width to be a tile multiple, it cannot be 16
            0);
    } else {  // auto-sharding
        std::optional<sliding_window::ParallelConfig> sw_parallel_config = pool::determine_pool_config_for_auto_shard(
            input_dtype,
            input_layout,
            core_grid,
            sliding_window_config,
            channels,
            pool_type,
            count_include_pad,
            divisor_override,
            return_indices,
            output_layout,
            output_dtype,
            config_tensor_in_dram);
        TT_FATAL(
            sw_parallel_config.has_value(),
            "autosharding could not determine valid shard scheme, please check tensor dimensions");
        parallel_config = sw_parallel_config.value();
    }

    uint32_t num_cores_c = conv::get_num_cores_channels_from_parallel_config(parallel_config);

    uint32_t input_tensor_width_snapped_to_channels_alignment =
        tt::round_up(channels, num_cores_c * input_channels_alignment);

    // Create target shape and apply sharding
    Shape input_tensor_shape = conv::flatten_4d_shape(input_shape);
    ttnn::Shape input_padded_shape = ttnn::Shape(
        {input_tensor_shape[0],
         input_tensor_shape[1],
         input_tensor_shape[2],
         input_tensor_width_snapped_to_channels_alignment});
    auto input_tensor_memory_config = conv::create_sharded_memory_config_from_parallel_config(
        input_padded_shape, parallel_config, is_in_tiled ? tt::constants::TILE_HEIGHT : 1);
    log_trace(
        tt::LogOp,
        "Pool Input = {}, Output = {}, Memory Config = {}",
        input_shape,
        output_shape,
        input_tensor_memory_config);
    return {input_tensor_memory_config, input_tensor_width_snapped_to_channels_alignment, parallel_config};
}

static std::vector<Tensor> pool2d_L1(
    const Tensor& input_tensor,
    Pool2DType pool_type,
    uint32_t batch_size,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::optional<std::array<uint32_t, 2>> dilation = std::nullopt,
    bool ceil_mode = false,
    bool count_include_pad = true,
    std::optional<int32_t> divisor_override = std::nullopt,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const TensorMemoryLayout> applied_shard_scheme = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    bool deallocate_input = false,
    bool reallocate_halo_output = true,
    bool return_indices = false,
    const DataType dtype = DataType::BFLOAT16,
    const Layout output_layout = Layout::ROW_MAJOR,
    std::optional<std::array<uint32_t, 2>> ceil_pad = std::nullopt,
    bool config_tensor_in_dram = false) {
    std::array<uint32_t, 4> padding_4d = sliding_window::get_pair_n4_padding(padding);
    bool is_out_tiled = output_layout == Layout::TILE;
    bool is_in_tiled = input_tensor.layout() == ttnn::TILE_LAYOUT;
    TT_FATAL(
        dtype == DataType::BFLOAT16 || dtype == DataType::BFLOAT8_B || dtype == DataType::BFLOAT4_B,
        "Currently only BFLOAT16, BFLOAT8_B, and BFLOAT4_B output data formats are supported");
    TT_FATAL(
        !((dtype == DataType::BFLOAT8_B || dtype == DataType::BFLOAT4_B) && output_layout == Layout::ROW_MAJOR),
        "BFLOAT8_B/BFLOAT4_B output data format is not supported with ROW_MAJOR layout");
    validate_input_params(
        input_tensor,
        batch_size,
        input_h,
        input_w,
        channels,
        kernel_size,
        stride,
        padding_4d[0],
        padding_4d[1],
        padding_4d[2],
        padding_4d[3],
        dilation.has_value() ? dilation.value()[0] : 1,
        dilation.has_value() ? dilation.value()[1] : 1,
        is_in_tiled);
    uint32_t dilation_h = dilation.has_value() ? dilation.value().at(0) : 1;
    uint32_t dilation_w = dilation.has_value() ? dilation.value().at(1) : 1;
    sliding_window::SlidingWindowConfig sliding_window_config{
        .batch_size = batch_size,
        .channels = channels,
        .input_hw = {input_h, input_w},
        .window_hw = {kernel_size.at(0), kernel_size.at(1)},
        .stride_hw = {stride.at(0), stride.at(1)},
        .padding = padding_4d,
        .dilation_hw = {dilation_h, dilation_w},
        .ceil_pad_hw = ceil_pad.has_value()
                           ? std::optional<sliding_window::uint32_pair_t>({ceil_pad->at(0), ceil_pad->at(1)})
                           : std::nullopt,
        .ceil_mode = ceil_mode,
    };
    auto output_shape = sliding_window_config.get_output_shape();
    const bool is_input_tensor_in_dram = input_tensor.memory_config().is_dram();
    sliding_window::ParallelConfig parallel_config;
    MemoryConfig out_memory_config = input_tensor.memory_config();
    uint32_t num_cores_nhw = 0;
    uint32_t num_cores_c = 0;
    Tensor input_tensor_sharded = input_tensor;
    if (!out_memory_config.shard_spec().has_value()) {
        // Input is not sharded. Perform sharding.

        ttnn::Shape input_tensor_shape = input_tensor.padded_shape();

        auto [in_memory_config, input_tensor_width_snapped_to_channels_alignment, calc_parallel_config] =
            get_pool_input_memory_config(
                sliding_window_config,
                applied_shard_scheme,
                batch_size,
                channels,
                input_tensor_shape,
                output_shape,
                input_tensor.device()->compute_with_storage_grid_size(),
                input_tensor.dtype(),
                input_tensor.layout(),
                dtype,
                output_layout,
                pool_type,
                count_include_pad,
                divisor_override,
                return_indices,
                config_tensor_in_dram);
        parallel_config = calc_parallel_config;
        bool is_tensor_already_flattened = (input_tensor_shape[0] == 1 && input_tensor_shape[1] == 1);
        Tensor input_tensor_flattened = input_tensor;
        // If tensor is in (n,h,w,c) format, flatten it to (1,1,nhw,c) for optimal sharding
        if (!is_tensor_already_flattened) {
            const auto flattened_input_shape = conv::flatten_4d_shape(input_tensor.logical_shape());
            const auto flattened_padded_input_shape = conv::flatten_4d_shape(input_tensor.padded_shape());
            input_tensor_flattened = ttnn::reshape(input_tensor, flattened_input_shape, flattened_padded_input_shape);
            input_tensor_shape = flattened_input_shape;
        }
        // Calculate padding needed for channels dimension
        uint32_t input_channels = input_tensor_shape[3];
        uint32_t padding_needed = input_tensor_width_snapped_to_channels_alignment - input_channels;

        // Apply zero padding to channels if needed - we need it in case when output dtype is block float because if we
        // have random values it would affect common exponent calculation

        Tensor input_tensor_padded;
        if (padding_needed > 0 && is_block_float(dtype)) {
            ttnn::SmallVector<std::array<uint32_t, 2>> pad_spec = {{0, 0}, {0, 0}, {0, 0}, {0, padding_needed}};

            input_tensor_padded = ttnn::pad(input_tensor, pad_spec, 0.0f);
        } else {
            input_tensor_padded = input_tensor;
        }
        input_tensor_sharded = ttnn::to_memory_config(input_tensor_flattened, in_memory_config, std::nullopt);
        out_memory_config = input_tensor_sharded.memory_config();

    } else {
        TT_FATAL(
            !applied_shard_scheme.has_value(), "A sharding scheme should not be specified for a sharded input tensor.");
        // input is already sharded, use it as is
        TT_FATAL(
            out_memory_config.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
            "Only row major orientation is supported.");

        parallel_config.grid = out_memory_config.shard_spec().value().grid;
        parallel_config.shard_scheme = out_memory_config.memory_layout();
        parallel_config.shard_orientation = out_memory_config.shard_spec().value().orientation;
    }
    num_cores_nhw = conv::get_num_cores_nhw_from_parallel_config(parallel_config);
    num_cores_c = conv::get_num_cores_channels_from_parallel_config(parallel_config);

    // update the shard spec to match the output shape
    auto shard_spec = out_memory_config.shard_spec().value();
    uint32_t output_nhw = output_shape[0] * output_shape[1] * output_shape[2];
    uint32_t output_nhw_padded =
        tt::round_up(output_nhw, num_cores_nhw * (is_out_tiled ? tt::constants::TILE_HEIGHT : 1));
    uint32_t output_shard_height_padded = output_nhw_padded / num_cores_nhw;
    uint32_t output_c = channels;
    uint32_t output_c_padded = tt::round_up(
        output_c, num_cores_c * (is_out_tiled ? tt::constants::TILE_WIDTH : tt::constants::TILE_WIDTH / 2));
    uint32_t output_shard_width_padded = output_c_padded / num_cores_c;
    log_debug(
        tt::LogOp,
        "output_nhw: {}, output_nhw_padded: {}, output_shard_height_padded: {}, output_shard_width_padded: {}",
        output_nhw,
        output_nhw_padded,
        output_shard_height_padded,
        output_shard_width_padded);
    out_memory_config = out_memory_config.with_shard_spec(tt::tt_metal::ShardSpec{
        shard_spec.grid, {output_shard_height_padded, output_shard_width_padded}, ShardOrientation::ROW_MAJOR});
    sliding_window_config = sliding_window::SlidingWindowConfig{
        .batch_size = batch_size,
        .channels = channels,
        .input_hw = {input_h, input_w},
        .window_hw = {kernel_size.at(0), kernel_size.at(1)},
        .stride_hw = {stride.at(0), stride.at(1)},
        .padding = {padding_4d.at(0), padding_4d.at(1), padding_4d.at(2), padding_4d.at(3)},
        .dilation_hw = {dilation_h, dilation_w},
        .ceil_pad_hw = ceil_pad.has_value()
                           ? std::optional<sliding_window::uint32_pair_t>({ceil_pad->at(0), ceil_pad->at(1)})
                           : std::nullopt,
        .num_cores_nhw = num_cores_nhw,
        .num_cores_c = num_cores_c,
        .core_range_set = parallel_config.grid,
        .snap_to_tile = is_out_tiled,
        .ceil_mode = ceil_mode,
    };

    // call the halo uop
    Tensor haloed_tensor = ttnn::halo(
        input_tensor_sharded,
        sliding_window_config,
        get_bf16_pool_init_value(pool_type),  // pad_val
        false,
        parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
        input_tensor_sharded.memory_config(),
        is_out_tiled,
        config_tensor_in_dram);

    if (deallocate_input || is_input_tensor_in_dram) {
        input_tensor_sharded.deallocate(/*force*/ true);
    }

    if (reallocate_halo_output) {
        haloed_tensor = ttnn::move(haloed_tensor);
    }

    // NOLINTBEGIN(bugprone-use-after-move)
    const uint32_t pre_allocate_size =
        haloed_tensor.device()->allocator()->get_statistics(tt::tt_metal::BufferType::L1).total_allocated_bytes;
    // NOLINTEND(bugprone-use-after-move)

    // call the pool2d uop
    std::vector<Tensor> output_tensors = ttnn::prim::pool2d(
        haloed_tensor,
        sliding_window_config,
        pool_type,
        dtype,
        output_layout,
        out_memory_config,
        compute_kernel_config,
        count_include_pad,
        divisor_override,
        return_indices,
        pre_allocate_size,
        config_tensor_in_dram);

    // format and return the result
    if (memory_config.has_value() && memory_config.value() != out_memory_config) {
        for (auto& output_tensor : output_tensors) {
            output_tensor = ttnn::to_memory_config(output_tensor, memory_config.value(), std::nullopt);
        }
    }

    if (return_indices) {
        TT_FATAL(
            output_tensors.size() == 2,
            "Expected two output tensors when return_indices is true, but got {}.",
            output_tensors.size());
        return output_tensors;
    }
    TT_FATAL(output_tensors.size() == 1, "Expected a single output tensor when return_indices is false.");
    return output_tensors;
}

class Pool2dSliceAttr : public ttnn::operations::op_slicing::OpSliceAttr {
    uint32_t batch_size;
    IOShape input_shape;
    uint32_t channels;
    std::array<uint32_t, 2> kernel_size;
    std::array<uint32_t, 2> stride;
    std::array<uint32_t, 4> padding_n4;
    std::array<uint32_t, 2> dilation;
    std::array<uint32_t, 2> ceil_pad{};
    sliding_window::SlidingWindowConfig sliding_window_config;
    IOShape output_shape;
    bool ceil_mode;
    bool count_include_pad;
    std::optional<int32_t> divisor_override;
    bool return_indices;
    Pool2DType pool_type;
    DataType dtype;
    TensorMemoryLayout shard_layout;
    Layout input_layout;
    Layout output_layout;
    std::optional<DeviceComputeKernelConfig> compute_kernel_config;
    bool config_tensor_in_dram;
    MeshDevice* device;

public:
    Pool2dSliceAttr(
        uint32_t batch_size,
        IOShape input_shape,
        uint32_t channels,
        std::array<uint32_t, 2> kernel_size,
        std::array<uint32_t, 2> stride,
        std::array<uint32_t, 4> padding_n4,
        std::array<uint32_t, 2> dilation,
        bool ceil_mode,
        bool count_include_pad,
        std::optional<int32_t> divisor_override,
        std::optional<const TensorMemoryLayout> applied_shard_scheme,
        bool return_indices,
        Pool2DType pool_type,
        DataType dtype,
        Layout input_layout,
        Layout output_layout,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config,
        bool config_tensor_in_dram,
        MeshDevice* device) :
        batch_size(batch_size),
        input_shape(input_shape),
        channels(channels),
        kernel_size(kernel_size),
        stride(stride),
        padding_n4(padding_n4),
        dilation(dilation),
        ceil_mode(ceil_mode),
        count_include_pad(count_include_pad),
        divisor_override(divisor_override),
        return_indices(return_indices),
        pool_type(pool_type),
        dtype(dtype),
        input_layout(input_layout),
        output_layout(output_layout),
        compute_kernel_config(compute_kernel_config),
        config_tensor_in_dram(config_tensor_in_dram),
        device(device) {
        shard_layout = applied_shard_scheme.value_or(TensorMemoryLayout::HEIGHT_SHARDED);
        sliding_window_config = sliding_window::SlidingWindowConfig{
            .batch_size = batch_size,
            .channels = channels,
            .input_hw = {std::get<0>(input_shape), std::get<1>(input_shape)},
            .window_hw = {kernel_size.at(0), kernel_size.at(1)},
            .stride_hw = {stride.at(0), stride.at(1)},
            .padding = padding_n4,
            .dilation_hw = {dilation.at(0), dilation.at(1)},
            .ceil_mode = ceil_mode,
        };
        auto full_output_shape = sliding_window_config.get_output_shape();
        this->output_shape = IOShape{full_output_shape[1], full_output_shape[2]};
        this->ceil_pad = {
            sliding_window_config.get_ceil_pad_hw().first, sliding_window_config.get_ceil_pad_hw().second};
    }

    std::tuple<std::tuple<IOShape, IOShape>, std::array<uint32_t, 4>, std::array<uint32_t, 2>, uint32_t>
    get_input_slice_and_padding(const IOShape& output_slice_start, const IOShape& output_slice_end) const {
        auto [output_slice_height_start, output_slice_width_start] = output_slice_start;
        auto [output_slice_height_end, output_slice_width_end] = output_slice_end;
        int input_slice_height_start = (output_slice_height_start * stride[0]) - padding_n4[0];
        int input_slice_height_end = ((output_slice_height_end - 1) * stride[0]) - padding_n4[0] +
                                     ((kernel_size[0] - 1) * (dilation[0] - 1)) + kernel_size[0];
        int input_slice_width_start = (output_slice_width_start * stride[1]) - padding_n4[2];
        int input_slice_width_end = ((output_slice_width_end - 1) * stride[1]) - padding_n4[2] +
                                    ((kernel_size[1] - 1) * (dilation[1] - 1)) + kernel_size[1];

        int pad_top = std::max<int>(0, -input_slice_height_start);
        int pad_bottom = std::max<int>(0, input_slice_height_end - std::get<0>(input_shape));
        int pad_left = std::max<int>(0, -input_slice_width_start);
        int pad_right = std::max<int>(0, input_slice_width_end - std::get<1>(input_shape));

        input_slice_height_start = std::max<int>(0, input_slice_height_start);
        input_slice_height_end = std::min<int>(std::get<0>(input_shape), input_slice_height_end);
        input_slice_width_start = std::max<int>(0, input_slice_width_start);
        input_slice_width_end = std::min<int>(std::get<1>(input_shape), input_slice_width_end);

        std::array<uint32_t, 2> this_ceil_pad = {0, 0};
        auto [output_height, output_width] = output_shape;
        if (output_slice_height_start == 0) {
            pad_top = padding_n4[0];
            input_slice_height_start = 0;
        }
        if (output_slice_height_end == output_height) {
            pad_bottom = padding_n4[1];
            input_slice_height_end = std::get<0>(input_shape);
            this_ceil_pad[0] = ceil_pad[0];
        }
        if (output_slice_width_start == 0) {
            pad_left = padding_n4[2];
            input_slice_width_start = 0;
        }
        if (output_slice_width_end == output_width) {
            pad_right = padding_n4[3];
            input_slice_width_end = std::get<1>(input_shape);
            this_ceil_pad[1] = ceil_pad[1];
        }
        uint32_t width_rounding_value = (output_layout == tt::tt_metal::Layout::TILE) ? tt::constants::TILE_HEIGHT : 1;
        uint32_t output_slice_width = output_slice_width_end - output_slice_width_start;
        uint32_t input_slice_width = input_slice_width_end - input_slice_width_start;
        if (output_slice_width % width_rounding_value != 0) {
            uint32_t additional_padded_width = width_rounding_value - (output_slice_width % width_rounding_value);
            log_trace(
                tt::LogOp,
                "Pool2d Slicing: Additional output width of {} added to the right side.",
                additional_padded_width);

            output_slice_width += additional_padded_width;
            pad_right = (output_slice_width - 1) * stride[1] - input_slice_width +
                        ((kernel_size[1] - 1) * (dilation[1] - 1)) + kernel_size[1];
        }
        return {
            {{input_slice_height_start, input_slice_width_start}, {input_slice_height_end, input_slice_width_end}},
            {pad_top, pad_bottom, pad_left, pad_right},
            this_ceil_pad,
            output_slice_width};
    }

    std::tuple<IOShape, IOShape> get_input_slice(
        const IOShape& output_slice_start, const IOShape& output_slice_end) const override {
        return std::get<0>(get_input_slice_and_padding(output_slice_start, output_slice_end));
    }

    uint32_t get_L1_usage(
        const IOShape& /*output_slice_start*/,
        const IOShape& /*output_slice_end*/,
        const op_slicing::Op2DSliceConfig& /*slice_config*/) const override {
        return 0;
    }

    tt::tt_metal::MemoryConfig get_input_memory_config(
        const IOShape& output_slice_start, const IOShape& output_slice_end) const override {
        auto [input_slice, this_slice_padding, this_ceil_pad, this_output_width] =
            get_input_slice_and_padding(output_slice_start, output_slice_end);
        auto [input_slice_start, input_slice_end] = input_slice;
        uint32_t input_slice_height = std::get<0>(input_slice_end) - std::get<0>(input_slice_start);
        uint32_t input_slice_width = std::get<1>(input_slice_end) - std::get<1>(input_slice_start);
        uint32_t output_slice_height = std::get<0>(output_slice_end) - std::get<0>(output_slice_start);

        uint32_t input_nhw_rounding_value =
            (input_layout == tt::tt_metal::Layout::TILE) ? tt::constants::TILE_HEIGHT : 1;
        uint32_t input_slice_nhw =
            tt::round_up(batch_size * input_slice_height * input_slice_width, input_nhw_rounding_value);
        auto sliced_input_tensor_memory_config = std::get<0>(get_pool_input_memory_config(
            sliding_window_config,
            shard_layout,
            batch_size,
            channels,
            ttnn::Shape({1, 1, input_slice_nhw, channels}),
            ttnn::Shape({batch_size, output_slice_height, this_output_width, channels}),
            device->compute_with_storage_grid_size(),
            dtype,
            input_layout,
            dtype,
            output_layout,
            pool_type,
            count_include_pad,
            divisor_override,
            return_indices,
            config_tensor_in_dram));

        return sliced_input_tensor_memory_config;
    }

    std::vector<ttnn::Tensor> run_L1_op(
        const ttnn::Tensor& sliced_input_tensor,
        const IOShape& output_slice_start,
        const IOShape& output_slice_end) override {
        auto [input_slice, this_slice_padding, this_ceil_pad, this_output_width] =
            get_input_slice_and_padding(output_slice_start, output_slice_end);
        auto [input_slice_start, input_slice_end] = input_slice;
        auto [input_slice_height_start, input_slice_width_start] = input_slice_start;
        auto [input_slice_height_end, input_slice_width_end] = input_slice_end;

        int input_slice_height = input_slice_height_end - input_slice_height_start;
        int input_slice_width = input_slice_width_end - input_slice_width_start;
        auto this_ceil_mode = ceil_mode;
        if (this_ceil_pad[0] > 0 || this_ceil_pad[1] > 0) {
            this_ceil_mode = true;
        }
        return pool2d_L1(
            sliced_input_tensor,
            pool_type,
            batch_size,
            input_slice_height,
            input_slice_width,
            channels,
            kernel_size,
            stride,
            this_slice_padding,
            dilation,
            this_ceil_mode,
            count_include_pad,
            divisor_override,
            std::nullopt,
            std::nullopt,
            compute_kernel_config,
            true, /* deallocate_input to save L1 */
            true, /* reallocate_halo_output to save L1 */
            return_indices,
            dtype,
            output_layout,
            this_ceil_pad,
            config_tensor_in_dram);
    }

    std::string name() const override { return "Pool2D"; }
};

static std::vector<Tensor> pool2d_DRAM(
    const Tensor& input_tensor,
    Pool2DType pool_type,
    uint32_t batch_size,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::optional<std::array<uint32_t, 2>> dilation_ = std::nullopt,
    bool ceil_mode = false,
    bool count_include_pad = true,
    std::optional<int32_t> divisor_override = std::nullopt,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Op2DSliceConfig>& dram_slice_config_ = std::nullopt,
    const std::optional<const TensorMemoryLayout> applied_shard_scheme = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    bool deallocate_input = false,
    bool reallocate_halo_output = true,
    bool return_indices = false,
    const DataType dtype = DataType::BFLOAT16,
    const Layout output_layout = Layout::ROW_MAJOR,
    bool config_tensor_in_dram = false) {
    if (!dram_slice_config_.has_value() ||
        dram_slice_config_->slice_type == op_slicing::Op2DSliceConfig::SliceType::L1_FULL ||
        dram_slice_config_->num_slices == 1) {
        return pool2d_L1(
            input_tensor,
            pool_type,
            batch_size,
            input_h,
            input_w,
            channels,
            kernel_size,
            stride,
            padding,
            dilation_,
            ceil_mode,
            count_include_pad,
            divisor_override,
            memory_config,
            applied_shard_scheme,
            compute_kernel_config,
            deallocate_input,
            reallocate_halo_output,
            return_indices,
            dtype,
            output_layout,
            std::nullopt,
            config_tensor_in_dram);
    }
    TT_FATAL(!return_indices, "DRAM pooling with return_indices=True is not supported yet.");
    std::array<uint32_t, 4> padding_4d = sliding_window::get_pair_n4_padding(padding);
    auto dilation = dilation_.value_or(std::array<uint32_t, 2>{1, 1});
    sliding_window::SlidingWindowConfig sliding_window_config{
        .batch_size = batch_size,
        .channels = channels,
        .input_hw = {input_h, input_w},
        .window_hw = {kernel_size.at(0), kernel_size.at(1)},
        .stride_hw = {stride.at(0), stride.at(1)},
        .padding = padding_4d,
        .dilation_hw = {dilation.at(0), dilation.at(1)},
        .ceil_mode = ceil_mode,
    };
    Tensor input_tensor_on_device = input_tensor;
    input_tensor_on_device = ttnn::to_memory_config(
        input_tensor_on_device,
        MemoryConfig{
            TensorMemoryLayout::INTERLEAVED,
            BufferType::DRAM,
        },
        std::nullopt);
    auto output_shape = sliding_window_config.get_output_shape();
    uint32_t output_height = output_shape[1];
    uint32_t output_width = output_shape[2];

    const auto unflattened_input_shape = ttnn::Shape{batch_size, input_h, input_w, channels};
    input_tensor_on_device = ttnn::reshape(input_tensor_on_device, unflattened_input_shape, unflattened_input_shape);
    TT_FATAL(input_tensor_on_device.memory_config().is_dram(), "Pool DRAM expects the input tensor to be in DRAM.");
    TT_FATAL(
        input_tensor_on_device.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input Tensor to Pool DRAM should be in Interleaved Memory Layout");

    Tensor dram_output_tensor = tt::tt_metal::create_device_tensor(
        TensorSpec(
            ttnn::Shape({batch_size, output_height, output_width, channels}),
            tt::tt_metal::TensorLayout(
                dtype,
                tt::tt_metal::PageConfig(output_layout),
                MemoryConfig{
                    TensorMemoryLayout::INTERLEAVED,
                    BufferType::DRAM,
                })),
        input_tensor_on_device.device());
    std::vector<std::reference_wrapper<Tensor>> output_tensors = {std::ref(dram_output_tensor)};
    // Currently return_indices is not supported for DRAM Max Pooling.
    Tensor dram_output_indices_tensor;
    if (return_indices) {
        dram_output_indices_tensor = tt::tt_metal::create_device_tensor(
            TensorSpec(
                ttnn::Shape({batch_size, output_height, output_width, channels}),
                tt::tt_metal::TensorLayout(
                    DataType::UINT16,
                    tt::tt_metal::PageConfig(output_layout),
                    MemoryConfig{
                        TensorMemoryLayout::INTERLEAVED,
                        BufferType::DRAM,
                    })),
            input_tensor_on_device.device());
        output_tensors.push_back(std::ref(dram_output_indices_tensor));
    }

    auto dram_slice_config = dram_slice_config_.value();
    TT_FATAL(dram_slice_config.num_slices > 0, "Number of slices must be greater than zero for DRAM slicing.");
    auto pool_slice_attr = Pool2dSliceAttr(
        batch_size,
        Pool2dSliceAttr::IOShape{input_h, input_w},
        channels,
        kernel_size,
        stride,
        sliding_window::get_pair_n4_padding(padding),
        dilation,
        ceil_mode,
        count_include_pad,
        divisor_override,
        applied_shard_scheme,
        return_indices,
        pool_type,
        dtype,
        input_tensor_on_device.layout(),
        output_layout,
        compute_kernel_config,
        config_tensor_in_dram,
        input_tensor_on_device.device());
    ttnn::operations::op_slicing::run_sliced_op(
        input_tensor_on_device, output_tensors, &pool_slice_attr, dram_slice_config);

    if (deallocate_input) {
        input_tensor_on_device.deallocate(true);
    }
    if (return_indices) {
        return {dram_output_tensor, dram_output_indices_tensor};
    }
    return {dram_output_tensor};
}

// Enum to represent the execution path for pool2d operations
enum class Pool2dExecutionPath {
    L1,   // Execute Pool using L1 memory
    DRAM  // Execute Pool using DRAM slicing
};

// Helper function to determine which pool2d execution path to take based on
// slice configuration and input tensor properties
Pool2dExecutionPath determine_pool2d_execution_path(
    const ttnn::Tensor& /*input_tensor*/, const std::optional<const Op2DSliceConfig>& slice_config) {
    // If slice config explicitly specifies DRAM slicing, use DRAM path
    if (slice_config.has_value() && slice_config->slice_type != Op2DSliceConfig::SliceType::L1_FULL) {
        return Pool2dExecutionPath::DRAM;
    }

    // Otherwise, use L1 path
    return Pool2dExecutionPath::L1;
}

static std::vector<Tensor> pool2d(
    const Tensor& input_tensor,
    Pool2DType pool_type,
    uint32_t batch_size,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::optional<std::array<uint32_t, 2>> dilation = std::nullopt,
    bool ceil_mode = false,
    bool count_include_pad = true,
    std::optional<int32_t> divisor_override = std::nullopt,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Op2DSliceConfig>& dram_slice_config = std::nullopt,
    const std::optional<const TensorMemoryLayout> applied_shard_scheme = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    bool deallocate_input = false,
    bool reallocate_halo_output = true,
    bool return_indices = false,
    const DataType dtype = DataType::BFLOAT16,
    const Layout output_layout = Layout::ROW_MAJOR,
    bool config_tensor_in_dram = false) {
    if (determine_pool2d_execution_path(input_tensor, dram_slice_config) == Pool2dExecutionPath::L1) {
        return pool2d_L1(
            input_tensor,
            pool_type,
            batch_size,
            input_h,
            input_w,
            channels,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            count_include_pad,
            divisor_override,
            memory_config,
            applied_shard_scheme,
            compute_kernel_config,
            deallocate_input,
            reallocate_halo_output,
            return_indices,
            dtype,
            output_layout,
            std::nullopt,
            config_tensor_in_dram);
    }
    return pool2d_DRAM(
        input_tensor,
        pool_type,
        batch_size,
        input_h,
        input_w,
        channels,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        count_include_pad,
        divisor_override,
        memory_config,
        dram_slice_config,
        applied_shard_scheme,
        compute_kernel_config,
        deallocate_input,
        reallocate_halo_output,
        return_indices,
        dtype,
        output_layout,
        config_tensor_in_dram);
}

std::vector<Tensor> MaxPool2DOp::invoke(
    const Tensor& input_tensor,
    uint32_t batch_size,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    bool ceil_mode,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<Op2DSliceConfig>& dram_slice_config,
    const std::optional<const TensorMemoryLayout> applied_shard_scheme,
    bool deallocate_input,
    bool reallocate_halo_output,
    bool return_indices,
    const DataType dtype,
    const Layout output_layout,
    bool config_tensor_in_dram) {
    return pool2d(
        input_tensor,
        Pool2DType::MAX_POOL2D,
        batch_size,
        input_h,
        input_w,
        channels,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        true,          // count_include_pad
        std::nullopt,  // divisor_override
        memory_config,
        dram_slice_config,
        applied_shard_scheme,
        std::nullopt,  // compute_kernel_config - not needed for max pool
        deallocate_input,
        reallocate_halo_output,
        return_indices,
        dtype,
        output_layout,
        config_tensor_in_dram);
}

Tensor AvgPool2DOp::invoke(
    const Tensor& input_tensor,
    uint32_t batch_size,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int32_t> divisor_override,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<Op2DSliceConfig>& dram_slice_config,
    const std::optional<const TensorMemoryLayout> applied_shard_scheme,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    bool deallocate_input,
    bool reallocate_halo_output,
    const DataType dtype,
    const Layout output_layout,
    bool config_tensor_in_dram) {
    auto result = pool2d(
        input_tensor,
        Pool2DType::AVG_POOL2D,
        batch_size,
        input_h,
        input_w,
        channels,
        kernel_size,
        stride,
        padding,
        std::nullopt,  // dilation
        ceil_mode,
        count_include_pad,
        divisor_override,
        memory_config,
        dram_slice_config,
        applied_shard_scheme,
        compute_kernel_config,
        deallocate_input,
        reallocate_halo_output,
        false,  // return_indices
        dtype,
        output_layout,
        config_tensor_in_dram);

    // Average pool always returns just the tensor, never indices
    return result.at(0);
}

}  // namespace ttnn::operations::pool
