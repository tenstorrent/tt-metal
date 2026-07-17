// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <functional>
#include <type_traits>
#include <utility>
#include <variant>

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <ttnn/tensor/layout/tensor_layout.hpp>
#include <ttnn/tensor/tensor_spec.hpp>

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "reshape.hpp"
#include "reshape_common.hpp"
#include "device/reshape_device_operation.hpp"

namespace ttnn::operations::data_movement {
namespace detail {

static uint32_t collapse_second_dim(const ttnn::Shape& shape) {
    TT_FATAL((shape.rank() != 0), "Can't collapse rank 0 tensor shape");
    uint32_t second_dim = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(shape.rank()) - 1; ++i) {
        second_dim = second_dim * shape[i];
    }
    return second_dim;
}

// Largest n in [1, max_n] with dim % n == 0 and (dim / n) % align == 0, else 0.
static uint32_t find_best_n_1d(uint32_t dim, uint32_t max_n, uint32_t align) {
    for (uint32_t n = max_n; n > 0; n--) {
        if (dim % n == 0 && (dim / n) % align == 0) {
            return n;
        }
    }
    return 0;
}

// Flatten PadValue to float for fill_implicit_tile_padding.
//   - uint32 + FLOAT32: caller-supplied uint32 is the bit pattern of the desired float, use bit_cast.
//   - Otherwise: caller-supplied value is a numeric integer or float, use static_cast.
// Note: integer pad values > 2^24 lose precision through this float bottleneck.
static float pad_value_as_float(const PadValue& pad_value, DataType dtype) {
    return std::visit(
        [dtype](auto v) -> float {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, uint32_t>) {
                if (dtype == DataType::FLOAT32) {
                    return std::bit_cast<float>(v);
                }
            }
            return static_cast<float>(v);
        },
        pad_value);
}

// True if the inner-2D of `shape` is not tile-aligned, i.e. the tiled output has implicit padding lanes.
static bool has_inner_2d_tile_padding(
    const ttnn::Shape& shape,
    uint32_t tile_height = tt::constants::TILE_HEIGHT,
    uint32_t tile_width = tt::constants::TILE_WIDTH) {
    TT_ASSERT(shape.rank() >= 2, "has_inner_2d_tile_padding requires rank >= 2");
    return (shape[-1] % tile_width != 0) || (shape[-2] % tile_height != 0);
}

// Pad logical shape to the given tile geometry. The shared compute_padded_shape() helper
// currently ignores its tile args and always pads to 32x32; use this for tiny-tile tensors.
static ttnn::Shape compute_padded_shape_for_tile(ttnn::Shape logical_shape, uint32_t tile_height, uint32_t tile_width) {
    if (logical_shape.rank() == 1) {
        logical_shape = ttnn::Shape({1, logical_shape[0]});
    }
    ttsl::SmallVector<uint32_t> output_shape_vec(logical_shape.rank());
    std::copy(logical_shape.cbegin(), logical_shape.cend(), output_shape_vec.begin());
    if (output_shape_vec.size() >= 1) {
        output_shape_vec[output_shape_vec.size() - 1] =
            tt::round_up(output_shape_vec[output_shape_vec.size() - 1], tile_width);
    }
    if (output_shape_vec.size() >= 2) {
        output_shape_vec[output_shape_vec.size() - 2] =
            tt::round_up(output_shape_vec[output_shape_vec.size() - 2], tile_height);
    }
    return ttnn::Shape(std::move(output_shape_vec));
}

// Returns a sharded output MemoryConfig, or INTERLEAVED if no valid grid exists.
// Callers must check is_sharded() before calling interleaved_to_sharded.
//
// `explicit_memory_config`: caller passed `memory_config` to the public API; honor
// a supplied BLOCK spec when tile-aligned and its grid covers the output.
//
// `input_shard_spec`: when `memory_config` is sharded but has no shard_spec, seed
// derivation from the input tensor's spec (layout should match the input).
MemoryConfig recompute_shard_spec_for_output(
    const MemoryConfig& memory_config,
    const TensorSpec& output_shape,
    bool explicit_memory_config = false,
    const std::optional<tt::tt_metal::ShardSpec>& input_shard_spec = std::nullopt) {
    // Auto-derive: caller asked for a sharded output but didn't pin a shard_spec. Seed
    // from input_shard_spec (typically the input tensor's) and re-enter. The caller's
    // layout and buffer_type are preserved; the full ShardSpec (grid, orientation, and
    // per-core shape) comes from the input and will be re-derived for the output shape.
    if (!memory_config.shard_spec().has_value() && input_shard_spec.has_value()) {
        MemoryConfig seeded{memory_config.memory_layout(), memory_config.buffer_type(), input_shard_spec};
        // Layout-only request: re-derive shard_spec for output_shape; skip BLOCK explicit override.
        return recompute_shard_spec_for_output(seeded, output_shape, /*explicit_memory_config=*/false, std::nullopt);
    }
    auto output_mem_config = memory_config;
    TT_FATAL(
        memory_config.shard_spec().has_value(),
        "ttnn.reshape: sharded output memory_config (layout={}) was supplied without a shard_spec, "
        "and no input_shard_spec is available (input is likely interleaved). Either provide a "
        "shard_spec on memory_config or omit memory_config to inherit it from the input tensor.",
        memory_config.memory_layout());
    const auto& source_shard_spec = memory_config.shard_spec().value();
    auto orientation = source_shard_spec.orientation;

    auto alignment = output_shape.page_config().get_recommended_shard_shape_alignment(output_shape.data_type());
    uint32_t align_h = alignment.size() >= 2 ? alignment[-2] : 1;
    uint32_t align_w = alignment.size() >= 1 ? alignment[-1] : 1;
    auto phys_h = output_shape.physical_shape().height();
    auto phys_w = output_shape.physical_shape().width();

    // Reject non-rectangular grids; bounding_box() would silently include extra cores.
    // TODO: search for a rectangular sub-grid inside the input before falling back.
    auto input_bbox = source_shard_spec.grid.bounding_box();
    if (input_bbox.size() != source_shard_spec.grid.num_cores()) {
        log_warning(
            tt::LogOp,
            "ttnn.reshape: shard grid is non-rectangular "
            "(bbox cores={}, grid cores={}); falling back to INTERLEAVED",
            input_bbox.size(),
            source_shard_spec.grid.num_cores());
        return MemoryConfig{TensorMemoryLayout::INTERLEAVED, memory_config.buffer_type()};
    }

    const auto start = input_bbox.start_coord;
    const uint32_t grid_x = input_bbox.grid_size().x;
    const uint32_t grid_y = input_bbox.grid_size().y;
    const auto layout = memory_config.memory_layout();

    // Caller-explicit override: keep the supplied spec when it tile-aligns and its
    // grid covers the output's physical extent. Lets callers preserve a layout their
    // downstream ops are tuned for, even when our derived sub-grid would be smaller.
    if (explicit_memory_config && layout == TensorMemoryLayout::BLOCK_SHARDED) {
        const auto& shard_shape = source_shard_spec.shape;
        uint32_t shard_h = shard_shape[0];
        uint32_t shard_w = shard_shape[1];
        bool is_rm = (orientation == ShardOrientation::ROW_MAJOR);
        uint32_t cores_h = is_rm ? grid_y : grid_x;
        uint32_t cores_w = is_rm ? grid_x : grid_y;
        bool tile_aligned = (align_h == 0 || shard_h % align_h == 0) && (align_w == 0 || shard_w % align_w == 0);
        bool covers_output = ((uint64_t)cores_h * shard_h >= phys_h) && ((uint64_t)cores_w * shard_w >= phys_w);
        if (tile_aligned && covers_output) {
            return memory_config;
        }
    }

    if (layout == TensorMemoryLayout::BLOCK_SHARDED) {
        bool is_rm = (orientation == ShardOrientation::ROW_MAJOR);
        uint32_t max_ny = is_rm ? grid_y : grid_x;
        uint32_t max_nx = is_rm ? grid_x : grid_y;
        uint32_t ny = find_best_n_1d(phys_h, max_ny, align_h);
        uint32_t nx = find_best_n_1d(phys_w, max_nx, align_w);

        // Unreachable when phys dims are aligned (n=1 always valid); kept as a safety net.
        if (ny == 0 || nx == 0) {
            log_warning(
                tt::LogOp,
                "ttnn.reshape: cannot find valid BLOCK_SHARDED grid for output "
                "(phys_h={}, phys_w={}, align_h={}, align_w={}); falling back to INTERLEAVED",
                phys_h,
                phys_w,
                align_h,
                align_w);
            return MemoryConfig{TensorMemoryLayout::INTERLEAVED, memory_config.buffer_type()};
        }

        CoreCoord new_end =
            is_rm ? CoreCoord{start.x + nx - 1, start.y + ny - 1} : CoreCoord{start.x + ny - 1, start.y + nx - 1};
        output_mem_config = output_shape.block_sharded(CoreRange{start, new_end}, orientation).memory_config();
    } else if (layout == TensorMemoryLayout::HEIGHT_SHARDED || layout == TensorMemoryLayout::WIDTH_SHARDED) {
        const bool is_height = (layout == TensorMemoryLayout::HEIGHT_SHARDED);
        const uint32_t num_cores = source_shard_spec.grid.num_cores();
        const uint32_t phys_dim = is_height ? phys_h : phys_w;

        // Preserve the input grid; height/width_sharded() rounds the per-core shape up
        // to tile alignment without changing the grid itself.
        output_mem_config = is_height ? output_shape.height_sharded(source_shard_spec.grid, orientation).memory_config()
                                      : output_shape.width_sharded(source_shard_spec.grid, orientation).memory_config();

        // Warn when each core's shard slot is much larger than the data slice it actually backs.
        // 4x threshold skips the benign 2x case (e.g. 16-row data in a 32-row tile slot) and
        // only flags genuinely over-padded shards (e.g. UFLD V2's ~6.4x case).
        constexpr uint32_t kOverpadWarnThreshold = 4;
        if (output_mem_config.shard_spec().has_value()) {
            const auto& out_shape = output_mem_config.shard_spec().value().shape;
            const uint32_t shard_dim = is_height ? out_shape[0] : out_shape[1];
            const uint32_t data_per_core = (phys_dim + num_cores - 1) / num_cores;
            if (data_per_core > 0 && shard_dim / data_per_core >= kOverpadWarnThreshold) {
                log_warning(
                    tt::LogOp,
                    "ttnn.reshape: {} output per-core shard ({} {}) is {}x its per-core data "
                    "slice ({} {}) for phys_{}={} on {} cores. Per-core L1 is dominated by "
                    "padding; consider a smaller core grid or INTERLEAVED.",
                    layout,
                    shard_dim,
                    is_height ? "rows" : "cols",
                    shard_dim / data_per_core,
                    data_per_core,
                    is_height ? "rows" : "cols",
                    is_height ? 'h' : 'w',
                    phys_dim,
                    num_cores);
            }
        }
    } else {
        TT_FATAL(
            false, "Unsupported memory layout {}: expected BLOCK_SHARDED, HEIGHT_SHARDED, or WIDTH_SHARDED", layout);
    }
    return output_mem_config;
}

ttnn::Tensor perform_reshape_on_2D_RM(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const MemoryConfig& memory_config,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    // RM kernel assumes linear page ordering; use s2i/i2s for sharded buffers.
    auto temp_tensor = tensor;
    auto intermediate_out_memory_config = memory_config;

    if (tensor.memory_config().is_sharded()) {
        TT_FATAL(!sub_core_grid.has_value(), "Sharded reshape does not support sub core grid specification\n");
        MemoryConfig temp_memory_config{TensorMemoryLayout::INTERLEAVED, tensor.memory_config().buffer_type()};
        temp_tensor = ttnn::sharded_to_interleaved(tensor, temp_memory_config, std::nullopt);
    }
    if (memory_config.is_sharded()) {
        intermediate_out_memory_config =
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, intermediate_out_memory_config.buffer_type()};
    }

    auto temp_tensor2 = ttnn::prim::reshape_view(
        temp_tensor, logical_shape, padded_shape, intermediate_out_memory_config, false, sub_core_grid);

    if (memory_config.is_sharded()) {
        TT_FATAL(!sub_core_grid.has_value(), "Sharded reshape does not support sub core grid specification\n");
        auto output_mem_config = recompute_shard_spec_for_output(
            memory_config,
            temp_tensor2.tensor_spec(),
            /*explicit_memory_config=*/false,
            tensor.memory_config().shard_spec());
        if (output_mem_config.is_sharded()) {
            return ttnn::interleaved_to_sharded(temp_tensor2, output_mem_config, std::nullopt);
        }
    }
    return temp_tensor2;
}

ttnn::Tensor fix_shape_and_perform_reshape_on_2D_RM(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim,
    const MemoryConfig& memory_config,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    // This function turns a RM 2D->MD into an equivalent 2D->2D conversion and then turns the 2D output back to MD
    // using a 0 cost view
    TT_FATAL((logical_shape.rank() != 0), "Can't do reshape to rank 0 tensor");
    const uint32_t logical_second_dim = collapse_second_dim(logical_shape);
    const uint32_t padded_second_dim = collapse_second_dim(padded_shape);
    return PerformView(
        perform_reshape_on_2D_RM(
            tensor,
            ttnn::Shape({logical_second_dim, logical_shape[-1]}),
            ttnn::Shape({padded_second_dim, padded_shape[-1]}),
            memory_config,
            sub_core_grid),
        logical_shape,
        padded_shape,
        tile_first_dim,
        tile_second_dim);
}

// Wrapper to turn the ND-> MD problem into 3D->3D for tiled and 2D->2D for Row Major
ttnn::Tensor reshape_rm(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const uint32_t tile_first_dim,
    const uint32_t tile_second_dim,
    const MemoryConfig& memory_config,
    const PadValue& /*pad_value*/,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    // This function turns ND -> MD into 2D->MD for row major and 3D->MD for tiled using a 0 cost view
    TT_FATAL((tensor.logical_shape().rank() != 0), "Can't do reshape from rank 0 tensor");
    TT_FATAL(tensor.layout() == ttnn::ROW_MAJOR_LAYOUT, "Wrong layout in `reshape_rm` `");

    const auto& tensor_logical_shape = tensor.logical_shape();
    const auto& tensor_padded_shape = tensor.padded_shape();
    const uint32_t logical_second_dim = collapse_second_dim(tensor_logical_shape);
    const uint32_t padded_second_dim = collapse_second_dim(tensor_padded_shape);

    // Call reshape with the equivalent data 2D Row Major input tensor
    return fix_shape_and_perform_reshape_on_2D_RM(
        PerformView(
            tensor,
            Shape({logical_second_dim, tensor_logical_shape[-1]}),
            Shape({padded_second_dim, tensor_padded_shape[-1]}),
            tile_first_dim,
            tile_second_dim),
        logical_shape,
        padded_shape,
        tile_first_dim,
        tile_second_dim,
        memory_config,
        sub_core_grid);
}
}  // namespace detail

ttnn::Tensor PerformView(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const uint32_t tile_first_dim = tt::constants::TILE_HEIGHT,
    const uint32_t tile_second_dim = tt::constants::TILE_WIDTH) {
    if (tensor.logical_shape() == logical_shape && tensor.padded_shape() == padded_shape) {
        return tensor;
    }
    if (logical_shape.rank() == 1) {
        return ttnn::experimental::view(tensor, logical_shape);
    }
    if (tensor.layout() == ttnn::TILE_LAYOUT &&
        (logical_shape[-1] % tile_second_dim != 0 || logical_shape[-2] % tile_first_dim != 0)) {
        return ttnn::experimental::view(
            tensor,
            logical_shape,
            detail::compute_padded_shape_for_tile(logical_shape, tile_first_dim, tile_second_dim));
    }
    // Perform a reshape (view)
    return ttnn::experimental::view(tensor, logical_shape, padded_shape);
}

std::pair<ttnn::Shape, ttnn::Shape> shape_corrector(
    const ttnn::Tensor& tensor, const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape) {
    // Correct the shape to account for inferred dimensions
    uint32_t input_volume = tensor.logical_volume();
    uint32_t output_volume = 1;
    uint32_t inferred_dim = -1;
    for (uint32_t i = 0; i < logical_shape.rank(); i++) {
        if ((static_cast<int>(logical_shape[i])) == -1) {
            if (inferred_dim != -1) {
                TT_FATAL(false, "Only one dimension can be inferred in reshape");
            }
            inferred_dim = i;
        } else {
            output_volume = output_volume * logical_shape[i];
        }
    }
    if (inferred_dim == -1) {
        return {logical_shape, padded_shape};
    }

    uint32_t implied_dim_value = (output_volume == 0) ? 0 : input_volume / output_volume;
    ttnn::Shape new_shape = logical_shape;
    new_shape[inferred_dim] = implied_dim_value;
    return {new_shape, new_shape};
}

ttnn::Tensor reshape_tiled(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_shape,
    const MemoryConfig& memory_config,
    const PadValue& pad_value,
    const bool recreate_mapping_tensor,
    const std::optional<CoreRangeSet>& sub_core_grid,
    bool explicit_memory_config,
    const bool skip_padding_fill,
    const bool pad_value_explicit) {
    // fill_implicit_tile_padding takes float; integer pad is not bit-exact for all dtypes/values.
    const float fill_value = detail::pad_value_as_float(pad_value, tensor.dtype());
    const auto& tile = tensor.tensor_spec().tile();
    const uint32_t tile_height = tile.get_height();
    const uint32_t tile_width = tile.get_width();
    // squeeze input tensor and requested shape to 3D
    auto transform_to_3d = [](const auto& shape) -> ttnn::Shape {
        if (shape.rank() > 3) {
            return squeeze_shape_to_3D(shape);
        }
        if (shape.rank() < 3) {
            return unsqueeze_shape_to_3D(shape);
        }
        return shape;
    };

    const auto input_tensor_shape_3d = transform_to_3d(tensor.logical_shape());
    const auto requested_shape_3d = transform_to_3d(logical_shape);

    const auto requested_padded_shape_3d =
        detail::compute_padded_shape_for_tile(requested_shape_3d, tile_height, tile_width);
    const auto input_padded_shape_3d =
        detail::compute_padded_shape_for_tile(input_tensor_shape_3d, tile_height, tile_width);
    auto tensor3d = PerformView(tensor, input_tensor_shape_3d, input_padded_shape_3d, tile_height, tile_width);

    if (memory_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
        memory_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
        memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        TT_FATAL(!sub_core_grid.has_value(), "Sharded reshape does not support sub core grid specification\n");

        // Block-float (BFLOAT8_B / BFLOAT4_B): typecast kernels require interleaved inputs, so keep
        // both typecasts on interleaved tensors and only convert to sharded at the very end.
        if (is_block_float(tensor.dtype())) {
            if (tensor.memory_config().is_sharded()) {
                MemoryConfig working_input_memory_config{
                    TensorMemoryLayout::INTERLEAVED, tensor.memory_config().buffer_type()};
                tensor3d = ttnn::sharded_to_interleaved(tensor3d, working_input_memory_config, std::nullopt);
            }
            tensor3d = ttnn::typecast(tensor3d, DataType::BFLOAT16);

            MemoryConfig working_output_memory_config = memory_config;
            if (memory_config.is_sharded()) {
                working_output_memory_config =
                    MemoryConfig{TensorMemoryLayout::INTERLEAVED, working_output_memory_config.buffer_type()};
            }
            auto output_tensor_3d = ttnn::prim::reshape_view(
                tensor3d,
                requested_shape_3d,
                requested_padded_shape_3d,
                working_output_memory_config,
                recreate_mapping_tensor,
                sub_core_grid);
            // Fill in BF16 (rank-3, non-recursive). skip_padding_fill is *intentionally*
            // ignored for block-float: shared exponent over 16-elem sub-blocks would otherwise
            // let unfilled padding corrupt logical lanes. See skip_padding_fill docstring.
            if (detail::has_inner_2d_tile_padding(requested_shape_3d, tile_height, tile_width)) {
                output_tensor_3d = ttnn::fill_implicit_tile_padding(output_tensor_3d, fill_value, std::nullopt);
            }
            output_tensor_3d = ttnn::typecast(output_tensor_3d, tensor.dtype());
            if (memory_config.is_sharded()) {
                auto output_mem_config = detail::recompute_shard_spec_for_output(
                    memory_config,
                    output_tensor_3d.tensor_spec(),
                    explicit_memory_config,
                    tensor.memory_config().shard_spec());
                if (output_mem_config.is_sharded()) {
                    output_tensor_3d = ttnn::interleaved_to_sharded(output_tensor_3d, output_mem_config, std::nullopt);
                }
            }
            return PerformView(
                output_tensor_3d,
                logical_shape,
                detail::compute_padded_shape_for_tile(logical_shape, tile_height, tile_width),
                tile_height,
                tile_width);
        }

        // Direct sharded path: TILED factories use TensorAccessorArgs for transparent sharded I/O.
        // Temporary interleaved TensorSpec provides physical dims and alignment to recompute_shard_spec_for_output.
        MemoryConfig target_output_mem_config = memory_config;
        if (memory_config.is_sharded()) {
            MemoryConfig interleaved_output_mem_config{TensorMemoryLayout::INTERLEAVED, memory_config.buffer_type()};
            auto interleaved_output_spec = TensorSpec(
                requested_shape_3d,
                tt::tt_metal::TensorLayout::fromPaddedShape(
                    tensor3d.dtype(),
                    tensor3d.tensor_spec().page_config(),
                    interleaved_output_mem_config,
                    requested_shape_3d,
                    requested_padded_shape_3d));
            target_output_mem_config = detail::recompute_shard_spec_for_output(
                memory_config, interleaved_output_spec, explicit_memory_config, tensor.memory_config().shard_spec());
        }

        // If recompute fell back to INTERLEAVED, un-shard the input so
        // prim::reshape_view sees matching layouts.
        if (tensor3d.memory_config().is_sharded() && !target_output_mem_config.is_sharded()) {
            MemoryConfig interleaved_input{TensorMemoryLayout::INTERLEAVED, tensor3d.memory_config().buffer_type()};
            tensor3d = ttnn::sharded_to_interleaved(tensor3d, interleaved_input, std::nullopt);
        }

        auto output_tensor_3d = ttnn::prim::reshape_view(
            tensor3d,
            requested_shape_3d,
            requested_padded_shape_3d,
            target_output_mem_config,
            recreate_mapping_tensor,
            sub_core_grid);

        // Fill implicit tile padding only when the caller explicitly asked for it (pad_value_explicit).
        // Default-on fill would write into "padding" lanes of prim::reshape_view, which can be a zero-cost
        // alias of the input buffer; in extreme low-rank cases (e.g. inner-2D (1, 1)) this clobbers
        // logical lanes of aliased tensors. Output may be sharded (typical) or INTERLEAVED if
        // recompute fell back; handle both. Tensors here are rank-3, so the fill stays
        // on its non-recursive path and cannot loop back into ttnn::reshape.
        if (pad_value_explicit && !skip_padding_fill &&
            detail::has_inner_2d_tile_padding(requested_shape_3d, tile_height, tile_width)) {
            if (output_tensor_3d.memory_config().is_sharded()) {
                // TODO(#43090): drop this s2i/i2s detour once prim::fill_pad supports sharded buffers
                // without overflowing fill_pad_writer's per-core runtime-arg cap (341).
                const auto sharded_mem_config = output_tensor_3d.memory_config();
                MemoryConfig interleaved_mem{TensorMemoryLayout::INTERLEAVED, sharded_mem_config.buffer_type()};
                auto interleaved = ttnn::sharded_to_interleaved(output_tensor_3d, interleaved_mem, std::nullopt);
                interleaved = ttnn::fill_implicit_tile_padding(interleaved, fill_value, std::nullopt);
                output_tensor_3d = ttnn::interleaved_to_sharded(interleaved, sharded_mem_config, std::nullopt);
            } else {
                output_tensor_3d = ttnn::fill_implicit_tile_padding(output_tensor_3d, fill_value, std::nullopt);
            }
        }

        return PerformView(
            output_tensor_3d,
            logical_shape,
            detail::compute_padded_shape_for_tile(logical_shape, tile_height, tile_width),
            tile_height,
            tile_width);
    }

    // Interleaved (DRAM / L1) tensors: call prim::reshape_view directly.
    if (is_block_float(tensor.dtype())) {
        TT_FATAL(!sub_core_grid.has_value(), "Block-float reshape does not support sub core grid specification\n");
        tensor3d = ttnn::typecast(tensor3d, DataType::BFLOAT16);
    }

    auto updated_mem_config = memory_config;
    // If block/height-sharded output, compute the correct shard spec
    if (updated_mem_config.is_sharded()) {
        // Synthesize TensorLayout from the requested padded shape, but with an interleaved
        // MemoryConfig (no shard_spec). Dropping the shard_spec is what lets the TensorSpec
        // constructor below skip its shape-fits-shard-grid validation, which would otherwise
        // apply the *input* shard_spec to the *output* shape and fatal. The padded shape
        // passed here ends up baked into the layout's alignment, so synthetic_spec.physical_shape()
        // exactly matches the requested padded shape, even if compute_padded_shape ever produced
        // dimensions that exceed tile alignment (e.g., due to shard-aware padding).
        auto synthetic_layout = tt::tt_metal::TensorLayout::fromPaddedShape(
            tensor3d.dtype(),
            tensor3d.tensor_spec().page_config(),
            MemoryConfig(updated_mem_config.buffer_type()),
            requested_shape_3d,
            requested_padded_shape_3d);

        // Construct synthetic TensorSpec
        tt::tt_metal::TensorSpec synthetic_spec(requested_shape_3d, synthetic_layout);

        // Recompute the shard spec for the output tensor shape
        updated_mem_config = detail::recompute_shard_spec_for_output(
            updated_mem_config, synthetic_spec, explicit_memory_config, tensor.memory_config().shard_spec());
    }

    auto output_tensor_3d = ttnn::prim::reshape_view(
        tensor3d,
        requested_shape_3d,
        requested_padded_shape_3d,
        updated_mem_config,
        recreate_mapping_tensor,
        sub_core_grid);

    // Fill rules:
    //   - Block-float (BFLOAT8_B / BFLOAT4_B): ALWAYS fill (skip_padding_fill is ignored). The downstream
    //     typecast's 16-elem shared exponent would otherwise let unfilled padding corrupt logical lanes in
    //     the same block.
    //   - Otherwise (BF16/FP32/int): fill only when the caller EXPLICITLY passed pad_value. A default-on
    //     fill writes into "padding" lanes of prim::reshape_view, which can be a zero-cost alias of the
    //     input buffer; in extreme low-rank cases (e.g. inner-2D (1, 1)) this clobbers logical lanes of
    //     aliased tensors and silently corrupts unrelated callers (e.g. tt-train AdamW / ColumnParallel).
    //   - skip_padding_fill (when caller did pass pad_value) still suppresses non-block-float fills.
    const bool is_block_float_output = is_block_float(tensor.dtype());
    const bool should_fill = is_block_float_output || (pad_value_explicit && !skip_padding_fill);
    if (should_fill && detail::has_inner_2d_tile_padding(requested_shape_3d, tile_height, tile_width)) {
        output_tensor_3d = ttnn::fill_implicit_tile_padding(output_tensor_3d, fill_value, std::nullopt);
    }

    if (is_block_float(tensor.dtype())) {
        TT_FATAL(!sub_core_grid.has_value(), "Block-float reshape does not support sub core grid specification\n");
        output_tensor_3d = ttnn::typecast(output_tensor_3d, tensor.dtype());
    }

    return PerformView(
        output_tensor_3d,
        logical_shape,
        detail::compute_padded_shape_for_tile(logical_shape, tile_height, tile_width),
        tile_height,
        tile_width);
}

}  // namespace ttnn::operations::data_movement

// Free function implementations
ttnn::Tensor ttnn::reshape(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& logical_input_shape,
    const ttnn::Shape& padded_input_shape,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<PadValue>& pad_value,
    const TileReshapeMapMode reshape_map_mode,
    const std::optional<CoreRangeSet>& sub_core_grid,
    const bool skip_padding_fill) {
    MemoryConfig mem_config = memory_config.value_or(tensor.memory_config());
    const bool explicit_memory_config = memory_config.has_value();
    auto layout = tensor.layout();
    auto tensor_shape = tensor.logical_shape();

    const auto [logical_shape, padded_shape] =
        operations::data_movement::shape_corrector(tensor, logical_input_shape, padded_input_shape);
    // First Case, No reshape Required
    if (tensor.logical_shape() == logical_shape && tensor.padded_shape() == padded_shape) {
        return tensor;
    }
    PadValue default_pad_value;
    if (tensor.dtype() == DataType::BFLOAT8_B or tensor.dtype() == DataType::BFLOAT4_B or
        tensor.dtype() == DataType::BFLOAT16 or tensor.dtype() == DataType::FLOAT32) {
        default_pad_value = 0.0f;
    } else {
        default_pad_value = (uint32_t)0;
    }

    const auto& tile = tensor.tensor_spec().tile();
    const uint32_t tile_first_dim = tile.get_height();
    const uint32_t tile_second_dim = tile.get_width();

    // The following case should only be called for the device storage case, the rest is a bandaid
    // for issue 15317

    const uint32_t shape_last_dim = logical_shape.rank() >= 1 ? logical_shape[-1] : 1;
    const uint32_t tensor_shape_last_dim = tensor_shape.rank() >= 1 ? tensor_shape[-1] : 1;
    const uint32_t shape_second_last_dim = logical_shape.rank() >= 2 ? logical_shape[-2] : 1;
    const uint32_t tensor_shape_second_last_dim = tensor_shape.rank() >= 2 ? tensor_shape[-2] : 1;

    // Just edit shape if shape has a 0 dimension
    if (tensor.logical_volume() == 0) {
        TT_FATAL(logical_shape.volume() == 0, "Tensor volume is 0, but shape's volume is not");
        return ttnn::experimental::view(tensor, logical_shape, padded_shape);
    }
    TT_FATAL(logical_shape.volume() != 0, "Tensor volume is not 0, but shape volume is 0");

    if (!is_device_tensor(tensor)) {
        // This case has been allowed in the past though it means introducing padding values to the data
        return ttnn::experimental::view(tensor, logical_shape, padded_shape);
    }

    bool this_is_view =
        (tensor_shape_last_dim == shape_last_dim) && (mem_config.is_sharded() == tensor.memory_config().is_sharded()) &&
        (mem_config.is_l1() == tensor.memory_config().is_l1()) &&
        ((tensor.layout() == ttnn::ROW_MAJOR_LAYOUT) ||              // Its row major
         (tensor_shape_second_last_dim == shape_second_last_dim) ||  // Second last dimension is the same
         (shape_second_last_dim % tile_first_dim == 0 &&
          tensor_shape_second_last_dim % tile_first_dim == 0));  // No tile-height padding on the second last dimension

    if (this_is_view) {
        return operations::data_movement::PerformView(
            tensor, logical_shape, padded_shape, tile_first_dim, tile_second_dim);
    }
    if (logical_shape.volume() != tensor.logical_volume()) {
        // This is completely incorrect but it is due to issue 15137 or issue 15558
        bool tile_tensor_view_reshape_possible =
            (layout == ttnn::Layout::TILE and padded_shape.rank() >= 2 and padded_shape[-2] % tile_first_dim == 0 and
             padded_shape[-1] % tile_second_dim == 0 and tensor.padded_shape()[-1] == padded_shape[-1]);

        if (tile_tensor_view_reshape_possible) {
            // This case has been allowed in the past though it means introducing padding values to the data
            return ttnn::experimental::view(tensor, logical_shape, padded_shape);
        }
        // This is a completely incorrect test but it is due to issue 15558
        TT_FATAL(false, "Attempting to reshape between two shapes with different volumes");
    }
    // Do the reshape in row-major
    if (tensor.layout() == ttnn::ROW_MAJOR_LAYOUT) {
        return operations::data_movement::detail::reshape_rm(
            tensor,
            logical_shape,
            padded_shape,
            tile_first_dim,
            tile_second_dim,
            mem_config,
            pad_value.value_or(default_pad_value),
            sub_core_grid);
    }
    // Preserve whether the caller explicitly passed pad_value. value_or(default_pad_value) below
    // collapses that signal, but reshape_tiled needs it to gate the default-off fill for non-BF8.
    const bool pad_value_explicit = pad_value.has_value();
    return operations::data_movement::reshape_tiled(
        tensor,
        logical_shape,
        mem_config,
        pad_value.value_or(default_pad_value),
        reshape_map_mode == TileReshapeMapMode::RECREATE,
        sub_core_grid,
        explicit_memory_config,
        skip_padding_fill,
        pad_value_explicit);
}

ttnn::Tensor ttnn::reshape(
    const ttnn::Tensor& tensor,
    const ttnn::Shape& shape,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<PadValue>& pad_value,
    const TileReshapeMapMode reshape_map_mode,
    const std::optional<CoreRangeSet>& sub_core_grid,
    const bool skip_padding_fill) {
    return reshape(tensor, shape, shape, memory_config, pad_value, reshape_map_mode, sub_core_grid, skip_padding_fill);
}

ttnn::Tensor ttnn::reshape(
    const ttnn::Tensor& tensor,
    ttsl::Span<const int32_t> shape_vector,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<PadValue>& pad_value,
    const TileReshapeMapMode reshape_map_mode,
    const std::optional<CoreRangeSet>& sub_core_grid,
    const bool skip_padding_fill) {
    return reshape(
        tensor,
        operations::data_movement::detail::infer_dims_for_reshape(tensor, shape_vector),
        memory_config,
        pad_value,
        reshape_map_mode,
        sub_core_grid,
        skip_padding_fill);
}
