// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation.hpp"

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/experimental/quasar/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/slice/slice.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/experimental/quasar/pad/pad.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"
#include "ttnn/operations/experimental/quasar/halo/halo.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/experimental/quasar/reshard/reshard.hpp"
#include "ttnn/operations/experimental/quasar/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "ttnn/operations/experimental/quasar/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/experimental/quasar/fold/device/fold_device_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/experimental/quasar/to_layout/to_layout_op.hpp"

#include "fold.hpp"

namespace ttnn::operations::experimental::quasar {

std::vector<Tensor> fold_with_transpose_(
    const Tensor& input,
    const std::optional<const ttnn::Shape>& output_shape,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t pad_c,
    uint32_t pad_h,
    uint32_t pad_w) {
    using namespace tt::constants;

    // Get the device
    if (input.storage_type() != StorageType::DEVICE) {
        TT_ASSERT(
            ttnn::GetDefaultDevice() != nullptr, "Requires setting default device if no inputs to op are on device");
    }

    uint32_t n = input.logical_shape()[0], c = input.logical_shape()[1], h = input.logical_shape()[2],
             w = input.logical_shape()[3];
    auto padded_c = c + pad_c;  // end padding only
    auto padded_h = h + pad_h;  // end padding
    auto padded_w = w + pad_w;  // end padding
    auto padded_h32 = tt::round_up(padded_h, TILE_HEIGHT);
    auto padded_w32 = tt::round_up(padded_w, TILE_HEIGHT);

    log_debug(tt::LogOp, "padded_c: {}", padded_c);
    log_debug(tt::LogOp, "padded_h: {}", padded_h);
    log_debug(tt::LogOp, "padded_w: {}", padded_w);
    log_debug(tt::LogOp, "padded_h32: {}", padded_h32);
    log_debug(tt::LogOp, "padded_w32: {}", padded_w32);

    auto L1_mem_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1};

    log_debug(tt::LogOp, "input: {}", input.logical_shape());

    // pad input tensor
    tt::tt_metal::Array4D padded_shape = {n, padded_c, padded_h32, padded_w32};
    auto pad_output =
        ttnn::operations::experimental::quasar::pad(input, padded_shape, tt::tt_metal::Array4D({0, 0, 0, 0}), 0);

    log_debug(tt::LogOp, "pad_output: {}", pad_output.logical_shape());

    auto transpose_hc_output = ttnn::prim::permute(
        pad_output, ttsl::SmallVector<uint32_t>({0, 3, 1, 2}), std::make_optional(L1_mem_config), std::nullopt);

    log_debug(tt::LogOp, "transpose_hc_output: {}", transpose_hc_output.logical_shape());

    // reshape
    n = transpose_hc_output.logical_shape()[0], w = transpose_hc_output.logical_shape()[1],
    c = transpose_hc_output.logical_shape()[2], h = transpose_hc_output.logical_shape()[3];
    auto reshape_hc_output =
        ttnn::reshape_on_device(transpose_hc_output, ttnn::Shape{n, (w / stride_w), (c * stride_w), h}, L1_mem_config);

    log_debug(tt::LogOp, "reshape_hc_output: {}", reshape_hc_output.logical_shape());

    // transpose
    auto transpose_hw_output2 =
        ttnn::operations::experimental::quasar::transpose(reshape_hc_output, 2, 3, L1_mem_config);

    log_debug(tt::LogOp, "transpose_hw_output2: {}", transpose_hw_output2.logical_shape());

    // reshape
    n = transpose_hw_output2.logical_shape()[0], w = transpose_hw_output2.logical_shape()[1],
    h = transpose_hw_output2.logical_shape()[2], c = transpose_hw_output2.logical_shape()[3];
    auto reshape_hw_output =
        ttnn::reshape_on_device(transpose_hw_output2, ttnn::Shape{n, w, (h / stride_h), (c * stride_h)}, L1_mem_config);

    log_debug(tt::LogOp, "reshape_hw_output: {}", reshape_hw_output.logical_shape());

    // transpose
    auto transpose_hc_output2 =
        ttnn::operations::experimental::quasar::transpose(reshape_hw_output, 1, 2, L1_mem_config);

    log_debug(tt::LogOp, "transpose_hc_output2: {}", transpose_hc_output2.logical_shape());

    std::vector<Tensor> output_tensors;
    if (output_shape.has_value()) {
        // slice
        n = output_shape.value()[0], w = output_shape.value()[1], h = output_shape.value()[2],
        c = output_shape.value()[3];
        tt::tt_metal::Array4D slice_output_tensor_start = {0, 0, 0, 0};
        tt::tt_metal::Array4D slice_output_tensor_end = {n, w, h, c};
        tt::tt_metal::Array4D step = {1, 1, 1, 1};
        auto slice_output = ttnn::operations::experimental::quasar::slice(
            transpose_hc_output2, slice_output_tensor_start, slice_output_tensor_end, step, L1_mem_config);

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

    // Get the device
    if (input.storage_type() != StorageType::DEVICE) {
        TT_ASSERT(
            ttnn::GetDefaultDevice() != nullptr, "Requires setting default device if no inputs to op are on device");
    }

    uint32_t n = input.logical_shape()[0], c = input.logical_shape()[1], h = input.logical_shape()[2],
             w = input.logical_shape()[3];
    auto padded_c = c + pad_c;        // end padding only
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
    auto tt_output_tensor = ttnn::operations::experimental::quasar::pad(
        input, padded_shape, tt::tt_metal::Array4D({0, 0, pad_h, 0}), 0, /*use_multicore*/ false, pad_mem_config);

    log_debug(tt::LogOp, "pad_output: {}", tt_output_tensor.logical_shape());

    // transpose
    tt_output_tensor = ttnn::operations::experimental::quasar::transpose(tt_output_tensor, 2, 3);

    log_debug(tt::LogOp, "transpose_hw_output: {}", tt_output_tensor.logical_shape());

    // pad tensor W dim
    tt::tt_metal::Array4D padded_shape2 = {n, padded_c, padded_h32, padded_w32};
    auto pad_mem_config2 = create_sharded_memory_config(ttnn::Shape(padded_shape2), grid_size, shard_spec.orientation);
    tt_output_tensor = ttnn::operations::experimental::quasar::pad(
        tt_output_tensor,
        padded_shape2,
        tt::tt_metal::Array4D({0, 0, pad_w, 0}),
        0,
        /*use_multicore*/ false,
        pad_mem_config2);

    log_debug(tt::LogOp, "pad_output: {}", tt_output_tensor.logical_shape());

    // transpose
    tt_output_tensor = ttnn::operations::experimental::quasar::transpose(tt_output_tensor, 1, 2);

    log_debug(tt::LogOp, "transpose_hc_output: {}", tt_output_tensor.logical_shape());

    // reshape
    n = tt_output_tensor.logical_shape()[0], w = tt_output_tensor.logical_shape()[1],
    c = tt_output_tensor.logical_shape()[2], h = tt_output_tensor.logical_shape()[3];
    tt_output_tensor = ttnn::experimental::view(tt_output_tensor, ttnn::Shape{n, (w / stride_w), (c * stride_w), h});

    log_debug(tt::LogOp, "reshape_hc_output: {}", tt_output_tensor.logical_shape());

    // transpose
    tt_output_tensor = ttnn::operations::experimental::quasar::transpose(tt_output_tensor, 2, 3);

    log_debug(tt::LogOp, "transpose_hw_output2: {}", tt_output_tensor.logical_shape());

    // reshape
    n = tt_output_tensor.logical_shape()[0], w = tt_output_tensor.logical_shape()[1],
    h = tt_output_tensor.logical_shape()[2], c = tt_output_tensor.logical_shape()[3];
    tt_output_tensor = ttnn::experimental::view(tt_output_tensor, ttnn::Shape{n, w, (h / stride_h), (c * stride_h)});

    log_debug(tt::LogOp, "reshape_hw_output: {}", tt_output_tensor.logical_shape());

    // transpose
    tt_output_tensor = ttnn::operations::experimental::quasar::transpose(tt_output_tensor, 1, 2);

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
        tt_output_tensor = ttnn::operations::experimental::quasar::slice(
            tt_output_tensor, slice_output_tensor_start, slice_output_tensor_end, steps, slice_mem_config);

        output_tensors.emplace_back(tt_output_tensor);

        log_debug(tt::LogOp, "slice_output: {}", tt_output_tensor.logical_shape());
    } else {
        // slice
        n = slice_output_shape[0], h = slice_output_shape[1], w = slice_output_shape[2], c = slice_output_shape[3];
        tt::tt_metal::Array4D slice_output_tensor_start = {0, 0, 0, 0};
        tt::tt_metal::Array4D slice_output_tensor_end = {n, h, w, c};
        auto slice_mem_config = create_sharded_memory_config(
            ttnn::Shape({n, h, w, c}), grid_size, shard_spec.orientation, override_memory_config);
        tt_output_tensor = ttnn::operations::experimental::quasar::slice(
            tt_output_tensor, slice_output_tensor_start, slice_output_tensor_end, steps, slice_mem_config);

        output_tensors.emplace_back(tt_output_tensor);

        log_debug(tt::LogOp, "slice_output: {}", tt_output_tensor.logical_shape());
    }

    return output_tensors;
}

// Extract padding values from variant
static std::array<uint32_t, 6> extract_padding_values(
    const std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>, std::array<uint32_t, 6>>& padding) {
    return std::visit(
        [](const auto& pad_array) -> std::array<uint32_t, 6> {
            using T = std::decay_t<decltype(pad_array)>;
            if constexpr (std::is_same_v<T, std::array<uint32_t, 2>>) {
                // [pad_h, pad_w] -> [pad_h, pad_h, pad_w, pad_w, 0, 0]
                return {pad_array[0], pad_array[0], pad_array[1], pad_array[1], 0, 0};
            } else if constexpr (std::is_same_v<T, std::array<uint32_t, 4>>) {
                // [pad_h_top, pad_h_bottom, pad_w_left, pad_w_right] -> [pad_h_top, pad_h_bottom, pad_w_left,
                // pad_w_right, 0, 0]
                return {pad_array[0], pad_array[1], pad_array[2], pad_array[3], 0, 0};
            } else {
                // [pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, pad_c_front, pad_c_back]
                return pad_array;
            }
        },
        padding);
}

// Helper function to validate height sharding
static void validate_height_sharding(const Tensor& tensor) {
    if (tensor.is_sharded() && tensor.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        TT_THROW("fold op does not support non height-sharding!");
    }
}

// Helper function to apply halo padding to sharded tensors
static Tensor apply_halo_padding(
    const Tensor& input_tensor, uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {
    using namespace ttnn::operations::sliding_window;

    auto input_shape = input_tensor.logical_shape();
    auto shard_spec = input_tensor.shard_spec().value();

    SlidingWindowConfig sliding_window_config{
        .batch_size = input_shape[0],
        .input_hw = {input_shape[1], input_shape[2]},
        .window_hw = {1, 1},
        .stride_hw = {1, 1},
        .padding = {pad_top, pad_bottom, pad_left, pad_right},
        .dilation_hw = {1, 1},
        .num_cores_nhw = static_cast<uint32_t>(shard_spec.grid.num_cores()),
        .num_cores_c = 1,
        .core_range_set = shard_spec.grid,
        .snap_to_tile = false};

    ttnn::Shape new_shape({1, 1, input_shape[0] * input_shape[1] * input_shape[2], input_shape[3]});
    auto reshaped_tensor = ttnn::operations::experimental::quasar::reshape(input_tensor, new_shape);

    const auto compute_kernel_config = ttnn::init_device_compute_kernel_config(
        tt::tt_metal::hal::get_arch(),
        std::nullopt,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/true,
        /*default_fp32_acc=*/reshaped_tensor.dtype() == DataType::FLOAT32,
        /*default_l1_acc=*/false);
    // The standard ttnn::halo builds a legacy DataMovementKernel (ProgramDescriptor path), which is
    // rejected on Quasar (Gen2 requires QuasarDataMovementKernel / Metal-2.0). Use the Metal-2.0 quasar
    // halo op there, matching the standard call's flags (pad_val=0, remote_read=false,
    // transpose_mcast=false, is_out_tiled=false). WH/BH keep the standard halo (unchanged).
    Tensor halo_output;
    if (tt::tt_metal::hal::get_arch() == tt::ARCH::QUASAR) {
        halo_output = ttnn::operations::experimental::quasar::halo(
            reshaped_tensor, sliding_window_config, compute_kernel_config, 0, false, false, false);
    } else {
        halo_output = ttnn::halo(reshaped_tensor, sliding_window_config, compute_kernel_config, 0, false, false, false);
    }

    // Reshape back to padded original dimensions
    ::ttnn::Shape padded_shape(
        {input_shape[0], input_shape[1] + pad_top + pad_bottom, input_shape[2] + pad_left + pad_right, input_shape[3]});
    return ttnn::operations::experimental::quasar::reshape(halo_output, padded_shape);
}

// Each core needs to process multiple of (stride_h * input_width) rows to ensure that
// the fold operation can be performed locally and do not need to read from remote cores.
// This function checks if the current shard height is divisible by (stride_h * input_width).
// If not, it calculates an optimal number of cores and corresponding shard height
// to enable efficient fold computation across the tensor dimensions.
Tensor reshard_if_needed(const Tensor& input, const uint32_t stride_h, const uint32_t /*stride_w*/) {
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

        auto new_mem_config =
            MemoryConfig(input.memory_config().memory_layout(), input.memory_config().buffer_type(), new_shard_spec);
        // need to reshard
        return ttnn::operations::experimental::quasar::reshard(input, new_mem_config, std::nullopt);
    }
    return input;
}

}  // namespace ttnn::operations::experimental::quasar

namespace ttnn::operations::experimental::quasar {

Tensor fold(
    const ttnn::Tensor& input_tensor_,
    uint32_t stride_h,
    uint32_t stride_w,
    bool use_transpose_as_fold,
    const std::optional<const ttnn::Shape>& output_shape,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>, std::array<uint32_t, 6>> padding,
    const std::optional<CoreRangeSet>& core_grid,
    const std::optional<MemoryConfig>& override_memory_config,
    bool input_is_nhwc) {
    // Extract padding values
    const std::array<uint32_t, 6> padding_values = operations::experimental::quasar::extract_padding_values(padding);
    const uint32_t pad_top = padding_values[0];
    const uint32_t pad_bottom = padding_values[1];
    const uint32_t pad_left = padding_values[2];
    const uint32_t pad_right = padding_values[3];
    const uint32_t pad_c_front = padding_values[4];
    const uint32_t pad_c_back = padding_values[5];
    const uint32_t pad_c = pad_c_back;  // For backward compatibility, typically only end padding
    const uint32_t pad_h = pad_top;     // Use top padding for symmetric case
    const uint32_t pad_w = pad_left;    // Use left padding for symmetric case
    const bool has_hw_padding = (pad_top | pad_bottom | pad_left | pad_right) != 0;
    const bool has_c_padding = (pad_c_front | pad_c_back) != 0;

    const Tensor& input_tensor = input_tensor_;
    TT_ASSERT(input_tensor.logical_shape().rank() == 4, "Fold op only supports 4D tensors");

    // Legacy transpose-based fold (TODO: remove when #29514 is solved)
    if (use_transpose_as_fold) {
        if (input_tensor.is_sharded()) {
            operations::experimental::quasar::validate_height_sharding(input_tensor);
            return operations::experimental::quasar::fold_with_transpose_sharded_(
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
        }
        return operations::experimental::quasar::fold_with_transpose_(
                   input_tensor, output_shape, stride_h, stride_w, pad_c, pad_h, pad_w)
            .at(0);
    }
    // Modern sharded tensor path (also the Quasar channels-last direct path).
    if (input_is_nhwc || (input_tensor.memory_config().is_l1() && input_tensor.is_sharded())) {
        // prim::qsr::fold needs NHWC (C last), height-sharded row-major, and Quasar row-major shards need
        // a 16B-aligned page width (bf16 -> C must be a multiple of 8). Common pipeline below: obtain an
        // INTERLEAVED NHWC `processed_tensor`, pad C up to 8 (aligned), interleaved_to_sharded, halo(H,W),
        // fold -> C*s^2 = 32, then slice each spatial group's real (C+pad_c) channels back down to match
        // the golden.
        const auto l1_interleaved =
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1};
        // Step-6 reshapes target DRAM interleaved: an interleaved target makes the quasar reshape skip the
        // final interleaved_to_sharded reshard (which would allocate the full folded tensor on one core and
        // OOM the already-packed 2-core L1), and keeps the large [N*Ho*Wo, groups*C_aligned] intermediates
        // in DRAM instead of L1.
        const auto dram_interleaved =
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};

        uint32_t real_c;
        CoreRangeSet in_grid;
        Tensor processed_tensor;
        if (input_is_nhwc) {
            // Quasar path: input is already channels-last [N,H,W,C] (C last), so we SKIP the NCHW->NHWC
            // transpose. That transpose has no Quasar kernel -- ttnn transpose(2,3) on interleaved-RM
            // routes to a legacy DataMovementKernel via prim::permute (kernel.hpp:382), see ~/fold_quasar.md.
            // Have the caller upload channels-last and set input_is_nhwc=true instead.
            real_c = static_cast<uint32_t>(input_tensor.logical_shape()[3]);
            in_grid = input_tensor.is_sharded() ? input_tensor.shard_spec().value().grid
                                                : core_grid.value();  // caller must pass grid_size when interleaved
            processed_tensor =
                input_tensor.is_sharded()
                    ? ttnn::operations::experimental::quasar::sharded_to_interleaved(input_tensor, l1_interleaved)
                    : input_tensor;
        } else {
            // NCHW input (WH/BH). Transpose to NHWC on-device via transpose(1,2) then transpose(2,3).
            // On Quasar this FATALs (no WH transpose kernel); pass input_is_nhwc=true with a
            // channels-last upload for Quasar.
            operations::experimental::quasar::validate_height_sharding(input_tensor);
            real_c = static_cast<uint32_t>(input_tensor.logical_shape()[1]);
            in_grid = input_tensor.shard_spec().value().grid;
            processed_tensor = ttnn::operations::experimental::quasar::transpose(input_tensor, 1, 2, l1_interleaved);
            processed_tensor =
                ttnn::operations::experimental::quasar::transpose(processed_tensor, 2, 3, l1_interleaved);
        }
        const uint32_t groups = stride_h * stride_w;
        uint32_t c_keep;
        uint32_t c_aligned;
        if (input_is_nhwc) {
            // The caller uploaded channels-last already padded to a 16B-aligned width (Quasar row-major
            // shards need bf16 width % 8 == 0). We do NOT pad C on device: the quasar pad op cannot inject
            // channel padding on Quasar -- its RM factory self-loops cb_pad on a DM kernel, which Gen2
            // rejects (program_spec.cpp:1309). So real_c is already the aligned width; recover the kept
            // (real + pad_c) channel count per group from output_shape.
            TT_FATAL(
                output_shape.has_value(),
                "Quasar channels-last fold (input_is_nhwc) requires output_shape to recover the kept channel count");
            c_aligned = real_c;                                                // already aligned by caller
            c_keep = static_cast<uint32_t>(output_shape.value()[3]) / groups;  // e.g. 16 / 4 = 4
        } else {
            c_keep = real_c + pad_c_front + pad_c_back;  // golden per-group channels (e.g. 3+1=4)
            c_aligned = tt::round_up(c_keep, 8u);        // 16B-aligned fold-input C width (8)

            // 2) Pad C up to the 16B-aligned width, while interleaved (no sub-16B row-major shard width).
            const auto s = processed_tensor.logical_shape();  // [N,H,W,real_c]
            const tt::tt_metal::Array4D padded_shape = {
                static_cast<uint32_t>(s[0]), static_cast<uint32_t>(s[1]), static_cast<uint32_t>(s[2]), c_aligned};
            processed_tensor = ttnn::operations::experimental::quasar::pad(
                processed_tensor, padded_shape, tt::tt_metal::Array4D({0, 0, 0, 0}), 0);
        }

        // 3) Interleaved -> height-sharded (over N*H*W, width = C_aligned) for the halo/fold path.
        //    (quasar::reshard is sharded->sharded; the permute+pad output is interleaved, so use
        //    interleaved_to_sharded to move it onto the shard grid.)
        processed_tensor = ttnn::operations::experimental::quasar::interleaved_to_sharded(
            processed_tensor,
            create_sharded_memory_config(processed_tensor.logical_shape(), in_grid, ShardOrientation::ROW_MAJOR));

        // 4) H,W halo padding (now correctly uses dims 1,2 = H,W).
        if (has_hw_padding) {
            processed_tensor = operations::experimental::quasar::apply_halo_padding(
                processed_tensor, pad_top, pad_bottom, pad_left, pad_right);
        }

        if (processed_tensor.layout() == Layout::TILE) {
            processed_tensor = ttnn::operations::experimental::quasar::to_layout(processed_tensor, Layout::ROW_MAJOR);
        }
        processed_tensor = operations::experimental::quasar::reshard_if_needed(processed_tensor, stride_h, stride_w);

        // 5) fold: prim::qsr::fold flattens (N,Ho,Wo) -> [1, 1, N*Ho*Wo, groups * C_aligned]. Capture the
        //    real NHWC output dims first (from the pre-fold [N, H_fold, W_fold, C_aligned]) so we can
        //    un-flatten to [N, Ho, Wo, C] at the end to match the NHWC golden.
        const auto pre_fold = processed_tensor.logical_shape();
        const uint32_t out_n = static_cast<uint32_t>(pre_fold[0]);
        const uint32_t out_ho = static_cast<uint32_t>(pre_fold[1]) / stride_h;
        const uint32_t out_wo = static_cast<uint32_t>(pre_fold[2]) / stride_w;
        auto folded = ttnn::prim::qsr::fold(processed_tensor, stride_h, stride_w);

        // 6a) Fast path (C=8 / padding-absorbed-into-conv-weights): when the consumer accepts the full
        //     aligned width (c_keep == c_aligned, i.e. output_shape's C == groups * C_aligned), there are no
        //     per-group padding channels to strip. Skip the reshape/slice/reshape un-weave (the slow DRAM
        //     tail: on the 2-core emulator that chain is minutes of row-major page movement) and just
        //     un-flatten the fold's [1,1,N*Ho*Wo, groups*C_aligned] row dim back to NHWC. This mirrors how
        //     the WH/BH transpose fold already keeps the aligned width and relies on the first conv's weights
        //     being folded to groups*C_aligned with zero pad channels (pad_and_fold_conv_filters_for_unity_
        //     stride at align_c == C_aligned), so the padding is harmless and the strip is unnecessary.
        if (c_keep == c_aligned) {
            return ttnn::operations::experimental::quasar::reshape(
                folded, ttnn::Shape{out_n, out_ho, out_wo, groups * c_aligned}, dram_interleaved);
        }

        // 6b) Keep only the real c_keep channels of each of the `groups` spatial sub-positions (the rest of
        //    C_aligned is padding). 4D-only: reinterpret each group as a separate W-row, slice the prefix,
        //    fold the groups back into C. Result last dim = groups * c_keep (e.g. 4*4 = 16), matching golden.
        const auto fs = folded.logical_shape();  // [N, Ho, Wo, groups * C_aligned]
        folded = ttnn::operations::experimental::quasar::reshape(
            folded,
            ttnn::Shape{
                static_cast<uint32_t>(fs[0]),
                static_cast<uint32_t>(fs[1]),
                static_cast<uint32_t>(fs[2]) * groups,
                c_aligned},
            dram_interleaved);
        folded = ttnn::operations::experimental::quasar::slice(
            folded,
            tt::tt_metal::Array4D({0, 0, 0, 0}),
            tt::tt_metal::Array4D(
                {static_cast<uint32_t>(fs[0]),
                 static_cast<uint32_t>(fs[1]),
                 static_cast<uint32_t>(fs[2]) * groups,
                 c_keep}),
            tt::tt_metal::Array4D({1, 1, 1, 1}),
            dram_interleaved);
        // Un-flatten the fold's [1,1,N*Ho*Wo, groups*c_keep] back to NHWC [N, Ho, Wo, groups*c_keep].
        folded = ttnn::operations::experimental::quasar::reshape(
            folded, ttnn::Shape{out_n, out_ho, out_wo, groups * c_keep}, dram_interleaved);
        return folded;
    }
    // Interleaved tensor path (DRAM or L1)
    Tensor processed_tensor = input_tensor;

    // Apply padding if needed
    if (has_hw_padding || has_c_padding) {
        ttsl::SmallVector<ttnn::operations::experimental::quasar::PadSpecDim> padding_spec;
        padding_spec.push_back({0, 0});                     // N dimension
        padding_spec.push_back({pad_top, pad_bottom});      // H dimension
        padding_spec.push_back({pad_left, pad_right});      // W dimension
        padding_spec.push_back({pad_c_front, pad_c_back});  // C dimension

        processed_tensor =
            ttnn::operations::experimental::quasar::pad(processed_tensor, padding_spec, 0.0f, true, std::nullopt);
    }

    const auto shape = processed_tensor.logical_shape();
    const auto batch_size = shape[0];
    const auto input_height = shape[1];
    const auto input_width = shape[2];
    const auto in_channels = shape[3];
    const bool was_tiled = processed_tensor.layout() == Layout::TILE;

    // The interleaved fold kernels operate on row-major data, so untilize first.
    if (was_tiled) {
        processed_tensor = ttnn::operations::experimental::quasar::to_layout(processed_tensor, Layout::ROW_MAJOR);
    }

    auto output_tensor = ttnn::prim::qsr::fold(processed_tensor, stride_h, stride_w);

    // Reshape output if input was tiled
    if (was_tiled) {
        const ttnn::Shape final_shape(
            {batch_size, input_height / stride_h, input_width / stride_w, in_channels * stride_h * stride_w});
        return ttnn::operations::experimental::quasar::reshape(output_tensor, final_shape);
    }

    return output_tensor;
}

}  // namespace ttnn::operations::experimental::quasar
