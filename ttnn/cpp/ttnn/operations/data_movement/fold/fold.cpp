// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// #include "ttnn/deprecated/tt_dnn/op_library/fold/fold_op.hpp"

#include "ttnn/run_operation.hpp"

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/sharded/reshard/reshard.hpp"

#include "ttnn/operations/experimental/reshape/view.hpp"

#include "fold.hpp"

namespace ttnn::operations::data_movement {

std::vector<Tensor> fold_with_transpose_(
    const Tensor& input,
    const std::optional<const ttnn::Shape>& output_shape,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t pad_c,
    uint32_t pad_h,
    uint32_t pad_w) {
    using namespace tt::constants;
    IDevice* device;

    // Get the device
    if (input.storage_type() != StorageType::DEVICE) {
        device = ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = input.device();
    }

    uint32_t n = input.logical_shape()[0], c = input.logical_shape()[1], h = input.logical_shape()[2],
             w = input.logical_shape()[3];
    auto padded_c = c + pad_c;  // end padding only
    auto padded_h = h + pad_h;  // end padding
    auto padded_w = w + pad_w;  // end padding
    auto padded_h32 = tt::round_up(padded_h, TILE_HEIGHT);
    auto padded_w32 = tt::round_up(padded_w, TILE_HEIGHT);

    log_info(tt::LogOp, "padded_c: {}", padded_c);
    log_info(tt::LogOp, "padded_h: {}", padded_h);
    log_info(tt::LogOp, "padded_w: {}", padded_w);
    log_info(tt::LogOp, "padded_h32: {}", padded_h32);
    log_info(tt::LogOp, "padded_w32: {}", padded_w32);

    auto L1_mem_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1};

    log_debug(tt::LogOp, "input: {}", input.logical_shape());

    // pad input tensor
    tt::tt_metal::Array4D padded_shape = {n, padded_c, padded_h32, padded_w32};
    auto pad_output = ttnn::pad(input, padded_shape, tt::tt_metal::Array4D({0, 0, 0, 0}), 0);

    log_debug(tt::LogOp, "pad_output: {}", pad_output.logical_shape());

    auto transpose_hc_output = ttnn::prim::permute(
        pad_output, ttnn::SmallVector<uint32_t>({0, 3, 1, 2}), std::make_optional(L1_mem_config), std::nullopt);

    log_debug(tt::LogOp, "transpose_hc_output: {}", transpose_hc_output.logical_shape());

    // reshape
    n = transpose_hc_output.logical_shape()[0], w = transpose_hc_output.logical_shape()[1],
    c = transpose_hc_output.logical_shape()[2], h = transpose_hc_output.logical_shape()[3];
    auto reshape_hc_output =
        ttnn::reshape_on_device(transpose_hc_output, ttnn::Shape{n, (w / stride_w), (c * stride_w), h}, L1_mem_config);

    log_debug(tt::LogOp, "reshape_hc_output: {}", reshape_hc_output.logical_shape());

    // transpose
    auto transpose_hw_output2 = ttnn::transpose(reshape_hc_output, 2, 3, L1_mem_config);

    log_debug(tt::LogOp, "transpose_hw_output2: {}", transpose_hw_output2.logical_shape());

    // reshape
    n = transpose_hw_output2.logical_shape()[0], w = transpose_hw_output2.logical_shape()[1],
    h = transpose_hw_output2.logical_shape()[2], c = transpose_hw_output2.logical_shape()[3];
    auto reshape_hw_output =
        ttnn::reshape_on_device(transpose_hw_output2, ttnn::Shape{n, w, (h / stride_h), (c * stride_h)}, L1_mem_config);

    log_debug(tt::LogOp, "reshape_hw_output: {}", reshape_hw_output.logical_shape());

    // transpose
    auto transpose_hc_output2 = ttnn::transpose(reshape_hw_output, 1, 2, L1_mem_config);

    log_debug(tt::LogOp, "transpose_hc_output2: {}", transpose_hc_output2.logical_shape());

    std::vector<Tensor> output_tensors;
    if (output_shape.has_value()) {
        // slice
        n = output_shape.value()[0], w = output_shape.value()[1], h = output_shape.value()[2],
        c = output_shape.value()[3];
        tt::tt_metal::Array4D slice_output_tensor_start = {0, 0, 0, 0};
        tt::tt_metal::Array4D slice_output_tensor_end = {n, w, h, c};
        tt::tt_metal::Array4D step = {1, 1, 1, 1};
        auto slice_output =
            ttnn::slice(transpose_hc_output2, slice_output_tensor_start, slice_output_tensor_end, step, L1_mem_config);

        output_tensors.emplace_back(slice_output);

        log_debug(tt::LogOp, "slice_output: {}", slice_output.logical_shape());
    } else {
        output_tensors.emplace_back(transpose_hc_output2);
    }

    return output_tensors;
}

ttnn::MemoryConfig create_sharded_memory_config(
    ttnn::Shape tensor_shape,
    const CoreRangeSet& grid_size,
    const ShardOrientation orientation,
    const std::optional<MemoryConfig>& override_memory_config = std::nullopt) {
    if (override_memory_config.has_value()) {
        return override_memory_config.value();
    }

    uint32_t total_cores = grid_size.num_cores();

    uint32_t tensor_height = tensor_shape[-2] * tensor_shape[-3] * tensor_shape[-4];
    uint32_t tensor_width = tensor_shape[-1];
    uint32_t shard_height = tt::div_up(tensor_height, total_cores);
    uint32_t shard_width = tensor_width;

    auto sharded_memory_config = ttnn::MemoryConfig{
        ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
        ttnn::BufferType::L1,
        tt::tt_metal::ShardSpec{grid_size, {shard_height, shard_width}, orientation}};

    log_debug(tt::LogOp, "sharded_memory_config: {}", sharded_memory_config);

    return sharded_memory_config;
}

std::vector<Tensor> fold_with_transpose_sharded_(
    const Tensor& input,
    const std::optional<const ttnn::Shape>& output_shape,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t pad_c,
    uint32_t pad_h,
    uint32_t pad_w,
    const CoreRangeSet& grid_size,
    const std::optional<MemoryConfig>& override_memory_config) {
    using namespace tt::constants;
    IDevice* device;

    // Get the device
    if (input.storage_type() != StorageType::DEVICE) {
        device = ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = input.device();
    }

    uint32_t n = input.logical_shape()[0], c = input.logical_shape()[1], h = input.logical_shape()[2],
             w = input.logical_shape()[3];
    auto padded_c = c + pad_c;      // end padding only
    auto padded_h = h + (pad_h * 2);  // front and end padding
    auto padded_w = w + (pad_w * 2);  // front and end padding
    auto padded_h32 = tt::round_up(padded_h, TILE_HEIGHT);
    auto padded_w32 = tt::round_up(padded_w, TILE_HEIGHT);
    auto target_h = padded_h / stride_h;
    auto target_w = padded_w / stride_w;
    auto target_c = padded_c * stride_h * stride_w;
    tt::tt_metal::Array4D slice_output_shape = {n, target_h, target_w, target_c};

    log_debug(tt::LogOp, "padded_c: {}", padded_c);
    log_debug(tt::LogOp, "padded_h: {}", padded_h);
    log_debug(tt::LogOp, "padded_w: {}", padded_w);
    log_debug(tt::LogOp, "padded_h32: {}", padded_h32);
    log_debug(tt::LogOp, "padded_w32: {}", padded_w32);

    log_debug(tt::LogOp, "input: {}", input.logical_shape());

    auto shard_spec = input.shard_spec().value();

    // pad input tensor
    tt::tt_metal::Array4D padded_shape = {n, padded_c, padded_h32, w};
    auto pad_mem_config = create_sharded_memory_config(ttnn::Shape(padded_shape), grid_size, shard_spec.orientation);
    auto tt_output_tensor = ttnn::pad(
        input, padded_shape, tt::tt_metal::Array4D({0, 0, pad_h, 0}), 0, /*use_multicore*/ false, pad_mem_config);

    log_debug(tt::LogOp, "pad_output: {}", tt_output_tensor.logical_shape());

    // transpose
    auto tphw_mem_config =
        create_sharded_memory_config(tt_output_tensor.logical_shape(), grid_size, shard_spec.orientation);
    tt_output_tensor = ttnn::transpose(tt_output_tensor, 2, 3, tphw_mem_config);

    log_debug(tt::LogOp, "transpose_hw_output: {}", tt_output_tensor.logical_shape());

    // pad tensor W dim
    tt::tt_metal::Array4D padded_shape2 = {n, padded_c, padded_h32, padded_w32};
    auto pad_mem_config2 = create_sharded_memory_config(ttnn::Shape(padded_shape2), grid_size, shard_spec.orientation);
    tt_output_tensor = ttnn::pad(
        tt_output_tensor,
        padded_shape2,
        tt::tt_metal::Array4D({0, 0, pad_w, 0}),
        0,
        /*use_multicore*/ false,
        pad_mem_config2);

    log_debug(tt::LogOp, "pad_output: {}", tt_output_tensor.logical_shape());

    // transpose
    auto tphc_mem_config =
        create_sharded_memory_config(tt_output_tensor.logical_shape(), grid_size, shard_spec.orientation);
    tt_output_tensor = ttnn::transpose(tt_output_tensor, 1, 2, tphc_mem_config);

    log_debug(tt::LogOp, "transpose_hc_output: {}", tt_output_tensor.logical_shape());

    // reshape
    n = tt_output_tensor.logical_shape()[0], w = tt_output_tensor.logical_shape()[1],
    c = tt_output_tensor.logical_shape()[2], h = tt_output_tensor.logical_shape()[3];
    tt_output_tensor = ttnn::experimental::view(tt_output_tensor, ttnn::Shape{n, (w / stride_w), (c * stride_w), h});

    log_debug(tt::LogOp, "reshape_hc_output: {}", tt_output_tensor.logical_shape());

    // transpose
    auto tphw_mem_config2 =
        create_sharded_memory_config(tt_output_tensor.logical_shape(), grid_size, shard_spec.orientation);
    tt_output_tensor = ttnn::transpose(tt_output_tensor, 2, 3, tphw_mem_config2);

    log_debug(tt::LogOp, "transpose_hw_output2: {}", tt_output_tensor.logical_shape());

    // reshape
    n = tt_output_tensor.logical_shape()[0], w = tt_output_tensor.logical_shape()[1],
    h = tt_output_tensor.logical_shape()[2], c = tt_output_tensor.logical_shape()[3];
    tt_output_tensor = ttnn::experimental::view(tt_output_tensor, ttnn::Shape{n, w, (h / stride_h), (c * stride_h)});

    log_debug(tt::LogOp, "reshape_hw_output: {}", tt_output_tensor.logical_shape());

    // transpose
    auto tphc_mem_config2 =
        create_sharded_memory_config(tt_output_tensor.logical_shape(), grid_size, shard_spec.orientation);
    tt_output_tensor = ttnn::transpose(tt_output_tensor, 1, 2, tphc_mem_config2);

    log_debug(tt::LogOp, "transpose_hc_output2: {}", tt_output_tensor.logical_shape());

    std::vector<Tensor> output_tensors;
    // override output shape
    auto steps = tt::tt_metal::Array4D({1, 1, 1, 1});
    if (output_shape.has_value()) {
        // slice
        n = output_shape.value()[0], h = output_shape.value()[1], w = output_shape.value()[2],
        c = output_shape.value()[3];
        tt::tt_metal::Array4D slice_output_tensor_start = {0, 0, 0, 0};
        tt::tt_metal::Array4D slice_output_tensor_end = {n, h, w, c};
        auto slice_mem_config = create_sharded_memory_config(
            ttnn::Shape({n, h, w, c}), grid_size, shard_spec.orientation, override_memory_config);
        tt_output_tensor =
            ttnn::slice(tt_output_tensor, slice_output_tensor_start, slice_output_tensor_end, steps, slice_mem_config);

        output_tensors.emplace_back(tt_output_tensor);

        log_debug(tt::LogOp, "slice_output: {}", tt_output_tensor.logical_shape());
    } else {
        // slice
        n = slice_output_shape[0], h = slice_output_shape[1], w = slice_output_shape[2], c = slice_output_shape[3];
        tt::tt_metal::Array4D slice_output_tensor_start = {0, 0, 0, 0};
        tt::tt_metal::Array4D slice_output_tensor_end = {n, h, w, c};
        auto slice_mem_config = create_sharded_memory_config(
            ttnn::Shape({n, h, w, c}), grid_size, shard_spec.orientation, override_memory_config);
        tt_output_tensor =
            ttnn::slice(tt_output_tensor, slice_output_tensor_start, slice_output_tensor_end, steps, slice_mem_config);

        output_tensors.emplace_back(tt_output_tensor);

        log_debug(tt::LogOp, "slice_output: {}", tt_output_tensor.logical_shape());
    }

    return output_tensors;
}

// Each core needs to process multiple of (stride_h * input_width) rows to ensure that
// the fold operation can be performed locally and do not need to read from remote cores.
// This function checks if the current shard height is divisible by (stride_h * input_width).
// If not, it calculates an optimal number of cores and corresponding shard height
// to enable efficient fold computation across the tensor dimensions.
Tensor reshard_if_needed(const Tensor& input, const uint32_t stride_h, const uint32_t stride_w) {
    ttnn::Shape input_shape = input.logical_shape();
    uint32_t input_width = input_shape[2];
    uint32_t pixels_per_compute_row = stride_h * input_width;
    uint32_t current_shard_height = input.shard_spec().value().shape[0];
    const CoreCoord& compute_grid_size = input.device()->compute_with_storage_grid_size();
    if (current_shard_height % pixels_per_compute_row != 0) {
        uint32_t max_cores = compute_grid_size.x * compute_grid_size.y;
        uint32_t total_height = input_shape[0] * input_shape[1] * input_shape[2];

        uint32_t num_cores = max_cores;
        while (num_cores > 0 && (total_height / pixels_per_compute_row) % num_cores != 0) {
            num_cores--;
        }

        TT_ASSERT(num_cores != 0, "Could not find suitable number of cores for resharing the input tensor");

        uint32_t optimal_shard_height = tt::round_up(total_height / num_cores, pixels_per_compute_row);

        auto new_shard_spec = input.shard_spec().value();
        new_shard_spec.shape[0] = optimal_shard_height;

        // Create new core range set using the calculated number of cores
        CoreRangeSet new_core_range = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_grid_size, true);
        new_shard_spec.grid = new_core_range;

        auto new_mem_config = input.memory_config().with_shard_spec(new_shard_spec);
        // need to reshard
        return ttnn::reshard(input, new_mem_config, std::nullopt);
    }
    return input;
}

Tensor FoldOperation::invoke(
    const ttnn::Tensor& input_tensor_,
    uint32_t stride_h,
    uint32_t stride_w,
    bool use_transpose_as_fold,
    const std::optional<const ttnn::Shape>& output_shape,
    uint32_t pad_c,
    uint32_t pad_h,
    uint32_t pad_w,
    const std::optional<CoreRangeSet>& core_grid,
    const std::optional<MemoryConfig>& override_memory_config) {
    Tensor input_tensor = input_tensor_;
    if (use_transpose_as_fold) {
        if (input_tensor.is_sharded()) {
            if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
                return fold_with_transpose_sharded_(
                           input_tensor,
                           output_shape,
                           stride_h,
                           stride_w,
                           pad_c,
                           pad_h,
                           pad_w,
                           core_grid.value_or(CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{1, 1}}}),
                           override_memory_config)
                    .at(0);
            } else {
                TT_THROW("fold op does not support non height-sharding!");
            }
        } else {
            return fold_with_transpose_(input_tensor, output_shape, stride_h, stride_w, pad_c, pad_h, pad_w).at(0);
        }
    }
    if (input_tensor.memory_config().is_l1() && input_tensor.is_sharded()) {
        if (input_tensor.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
            TT_THROW("fold op does not support non height-sharding!");
        }
        if (pad_h != 0 || pad_w != 0 || pad_c != 0) {
            TT_THROW("Padding is not supported for L1 sharded folding");
        }
        auto sharded_input_tensor = reshard_if_needed(input_tensor, stride_h, stride_w);
        return ttnn::prim::fold(sharded_input_tensor, stride_h, stride_w, output_shape, 0, 0, 0);
    }
    if (input_tensor.memory_config().is_dram()) {
        if (pad_h != 0 || pad_w != 0 || pad_c != 0) {
            TT_THROW("Padding is not supported for DRAM folding");
        }
        auto batch_size = input_tensor.logical_shape()[0];
        auto input_height = input_tensor.logical_shape()[1];
        auto input_width = input_tensor.logical_shape()[2];
        auto in_channels = input_tensor.logical_shape()[3];
        auto fold_input_tensor = input_tensor;
        if (in_channels % 32 == 0 && fold_input_tensor.layout() == Layout::TILE) {
            // Convert to row-major layout for 32-channel aligned tensors to leverage faster untilize+RM fold path
            fold_input_tensor = ttnn::to_layout(input_tensor, Layout::ROW_MAJOR);
        }

        auto output_tensor = ttnn::prim::fold(fold_input_tensor, stride_h, stride_w, output_shape, pad_c, pad_h, pad_w);
        if (fold_input_tensor.layout() == Layout::TILE) {
            return ttnn::reshape(
                output_tensor,
                ttnn::Shape(
                    {batch_size, input_height / stride_h, input_width / stride_w, (in_channels)*stride_h * stride_w}));
        }
        return output_tensor;
    }
    return ttnn::prim::fold(input_tensor, stride_h, stride_w, output_shape, pad_c, pad_h, pad_w);
}

}  // namespace ttnn::operations::data_movement
