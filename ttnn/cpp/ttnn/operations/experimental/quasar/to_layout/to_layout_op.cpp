// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "to_layout_op.hpp"

#include <bit>
#include <limits>
#include <string_view>

#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/experimental/quasar/pad/pad.hpp"
#include "ttnn/operations/experimental/quasar/tilize/tilize.hpp"
#include "ttnn/operations/experimental/quasar/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/experimental/quasar/untilize/untilize.hpp"
#include "ttnn/operations/experimental/quasar/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/reshape.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::CMAKE_UNIQUE_NAMESPACE {
namespace {

tt::tt_metal::Tile resolve_effective_tile(
    const ttnn::Tensor& tensor, ttnn::Layout layout, const std::optional<tt::tt_metal::Tile>& requested_tile) {
    TT_FATAL(layout == Layout::TILE, "Effective tile is only defined for TILE conversions");
    if (tensor.layout() == Layout::TILE) {
        return requested_tile.value_or(tensor.tensor_spec().tile());
    }
    return requested_tile.value_or(tt::tt_metal::Tile{});
}

void validate_tile_semantics(
    const ttnn::Tensor& tensor, ttnn::Layout layout, const std::optional<tt::tt_metal::Tile>& requested_tile) {
    if (layout == Layout::ROW_MAJOR) {
        TT_FATAL(
            !requested_tile.has_value(),
            "ttnn::experimental::quasar::to_layout: tile argument is only supported when converting to TILE_LAYOUT");
        return;
    }

    if (layout == Layout::TILE && tensor.layout() == Layout::TILE && requested_tile.has_value()) {
        TT_FATAL(
            tensor.tensor_spec().tile() == requested_tile.value(),
            "ttnn::experimental::quasar::to_layout: TILE tensor already uses tile {}, cannot convert to tile {} "
            "without retilize",
            tensor.tensor_spec().tile(),
            requested_tile.value());
    }
}

bool requires_padding_change(const ttnn::Tensor& tensor, ttnn::Layout layout, const tt::tt_metal::Tile& target_tile) {
    if (layout == Layout::ROW_MAJOR) {
        // There shouldn't be extra paddings for Row Major layout
        return tensor.logical_shape() != tensor.padded_shape();
    }
    // It's okay for conversion to tile layout to preserve arbitrary padding as long as it satisfies the alignment
    tt::tt_metal::PageConfig page_config = tt::tt_metal::PageConfig(layout, target_tile);

    // Padded shape only (dtype-independent). Use TensorLayout, not a TensorSpec: TensorSpec rejects
    // FP8_E4M3 + TILE (fp8 is ROW_MAJOR-only) though fp8 is a valid tilize input.
    const auto padded_shape = tt::tt_metal::TensorLayout(tensor.dtype(), page_config, tensor.memory_config())
                                  .compute_padded_shape(tensor.padded_shape());
    return tensor.padded_shape() != padded_shape;
}

bool is_allowed_row_major_dtype(ttnn::DataType tensor_dtype, std::optional<ttnn::DataType> requested_dtype) {
    if (!requested_dtype.has_value() || requested_dtype.value() == tensor_dtype) {
        return true;
    }
    // untilize / untilize_with_unpadding convert BFLOAT8_B -> BFLOAT16 natively as part of de-tiling.
    return tensor_dtype == ttnn::DataType::BFLOAT8_B && requested_dtype.value() == ttnn::DataType::BFLOAT16;
}

constexpr std::string_view kRowMajorDtypeErrorMessage =
    "dtype cannot be different from tensor dtype when converting to ROW_MAJOR_LAYOUT on device "
    "(allowed exception: BFLOAT8_B -> BFLOAT16, which untilize handles natively)!";

Tensor to_layout_impl(
    const ttnn::Tensor& tensor_arg,
    const ttnn::Layout layout,
    const std::optional<ttnn::DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const float pad_value,
    const std::optional<tt::tt_metal::Tile>& tile) {
    validate_tile_semantics(tensor_arg, layout, tile);

    if (tensor_arg.layout() == layout) {
        if (dtype.has_value() and dtype.value() != tensor_arg.dtype()) {
            log_warning(
                tt::LogOp,
                "ttnn::to_layout: dtype is specified but the tensor is already in the requested layout! So, the "
                "dtype "
                "won't be changed!");
        }
        if (memory_config.has_value() and memory_config.value() != get_memory_config(tensor_arg).value()) {
            log_warning(
                tt::LogOp,
                "ttnn::to_layout: memory_config is specified but the tensor is already in the requested layout! "
                "So, "
                "the memory_config won't be changed!");
        }
        return tensor_arg;
    }

    const std::set<ttnn::Layout> supported_layouts = {
        ttnn::ROW_MAJOR_LAYOUT,
        ttnn::TILE_LAYOUT,
    };

    if (!supported_layouts.contains(layout)) {
        TT_THROW("ttnn::to_layout: Unsupported layout conversion from {} to {}!", tensor_arg.layout(), layout);
    }

    auto tensor = tensor_arg;
    auto output_shape = tensor_arg.logical_shape();
    auto output_memory_config =
        memory_config.value_or(ttnn::get_memory_config(tensor).value_or(ttnn::DRAM_MEMORY_CONFIG));

    const auto effective_tile =
        layout == Layout::TILE ? resolve_effective_tile(tensor_arg, layout, tile) : tt::tt_metal::Tile{};
    tt::tt_metal::PageConfig page_config = tt::tt_metal::PageConfig(Layout::TILE, effective_tile);
    // Padded shape only (dtype-independent). Use TensorLayout, not a TensorSpec: TensorSpec rejects
    // FP8_E4M3 + TILE (fp8 is ROW_MAJOR-only) though fp8 is a valid tilize input; the real output dtype
    // flows through `dtype` into tilize()/untilize() below.
    auto padded_output_shape = tt::tt_metal::TensorLayout(tensor_arg.dtype(), page_config, output_memory_config)
                                   .compute_padded_shape(tensor_arg.logical_shape());
    auto original_rank = tensor_arg.logical_shape().rank();
    const auto& original_shape = tensor_arg.logical_shape();

    if (layout == ttnn::TILE_LAYOUT) {
        if (tensor.padded_shape().size() < 2) {
            TT_FATAL(
                !tensor.is_sharded(),
                "ttnn::to_layout: Cannot convert a sharded device tensor with rank {} to TILE_LAYOUT. "
                "Tilize requires shard dimensions divisible by tile size, but rank promotion to 2D "
                "produces a shard height of 1. Move to interleaved first, then tilize.",
                tensor.padded_shape().size());
            const bool is_scalar = tensor.padded_shape().size() == 0;
            ttsl::SmallVector<uint32_t> new_padded_shape =
                is_scalar ? ttsl::SmallVector<uint32_t>{1, 1}
                          : ttsl::SmallVector<uint32_t>{1, tensor.padded_shape()[-1]};
            tensor = ttnn::experimental::view(tensor, tensor.logical_shape(), Shape(new_padded_shape));
        }
    }

    if (tt::tt_metal::is_device_tensor(tensor_arg)) {
        bool use_multicore_untilize = true;
        bool use_multicore_tilize = true;

        if (not requires_padding_change(tensor, layout, effective_tile)) {
            if (layout == ttnn::ROW_MAJOR_LAYOUT) {
                TT_FATAL(is_allowed_row_major_dtype(tensor_arg.dtype(), dtype), "{}", kRowMajorDtypeErrorMessage);
                TT_FATAL(
                    tensor.tensor_spec().tile() == tt::tt_metal::Tile{},
                    "ttnn::experimental::quasar::to_layout: device untilize only supports the default tile in this PR");
                return ttnn::operations::experimental::quasar::untilize(
                    tensor, output_memory_config, use_multicore_untilize, sub_core_grids);
            }
            if (layout == ttnn::TILE_LAYOUT) {
                TT_FATAL(
                    effective_tile == tt::tt_metal::Tile{},
                    "ttnn::experimental::quasar::to_layout: device tilize only supports the default tile in this PR");
                if (tensor.is_sharded()) {
                    uint32_t tile_height = effective_tile.get_height();
                    uint32_t tile_width = effective_tile.get_width();
                    const auto mem_config = get_memory_config(tensor).value();
                    uint32_t shard_h, shard_w;
                    if (mem_config.shard_spec().has_value()) {
                        shard_h = mem_config.shard_spec().value().shape[0];
                        shard_w = mem_config.shard_spec().value().shape[1];
                    } else {
                        const auto& nd_spec = mem_config.nd_shard_spec().value();
                        shard_h = nd_spec.shard_shape[-2];
                        shard_w = nd_spec.shard_shape[-1];
                    }
                    if (shard_h % tile_height != 0 or shard_w % tile_width != 0) {
                        TT_THROW(
                            "ttnn::to_layout: Sharded tensor must have shard shape that is a multiple of "
                            "TILE_SIZE!");
                    }
                }
                return ttnn::operations::experimental::quasar::tilize(
                    tensor,
                    output_memory_config,
                    dtype,
                    use_multicore_tilize,
                    false /* low perf mode */,
                    sub_core_grids,
                    effective_tile);
            }
            throw std::runtime_error("ttnn::to_layout: Unsupported layout!");
        }
        if (layout == ttnn::ROW_MAJOR_LAYOUT) {
            TT_FATAL(is_allowed_row_major_dtype(tensor_arg.dtype(), dtype), "{}", kRowMajorDtypeErrorMessage);
            TT_FATAL(
                tensor.tensor_spec().tile() == tt::tt_metal::Tile{},
                "ttnn::experimental::quasar::to_layout: device untilize only supports the default tile in this PR");

            if (tensor.is_sharded()) {
                output_memory_config =
                    memory_config.value_or(ttnn::get_memory_config(tensor).value_or(ttnn::DRAM_MEMORY_CONFIG));
            }
            Shape output_tensor_end(ttsl::SmallVector<uint32_t>(tensor.logical_shape().rank(), 0));
            int logical_rank = tensor.logical_shape().rank();
            for (int index = -1; index >= -logical_rank; --index) {
                output_tensor_end[index] = tensor.logical_shape()[index] - 1;
            }
            return ttnn::operations::experimental::quasar::untilize_with_unpadding(
                tensor, output_tensor_end, output_memory_config, use_multicore_untilize, sub_core_grids);
        }
        if (layout == ttnn::TILE_LAYOUT) {
            TT_FATAL(
                effective_tile == tt::tt_metal::Tile{},
                "ttnn::experimental::quasar::to_layout: device tilize only supports the default tile in this PR");
            if (tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
                // ttnn::tilize_with_val_padding doesn't support height sharded tensors
                // workaround by applying padding and then tilizing
                ttsl::SmallVector<std::array<uint32_t, 2>> padding(tensor.logical_shape().rank(), {0, 0});
                padding[padding.size() - 2] = {0, padded_output_shape[-2] - output_shape[-2]};
                padding[padding.size() - 1] = {0, padded_output_shape[-1] - output_shape[-1]};
                TT_FATAL(!sub_core_grids.has_value(), "Pad OP does not currently support sub core grid");
                tensor = ttnn::operations::experimental::quasar::pad(tensor, padding, pad_value, true, std::nullopt);
                return ttnn::operations::experimental::quasar::tilize(
                    tensor, output_memory_config, dtype, use_multicore_tilize, false, std::nullopt, effective_tile);
            } else {
                PadValue pad_value_variant;
                if (tensor.dtype() == ttnn::DataType::BFLOAT16 or tensor.dtype() == ttnn::DataType::FLOAT32) {
                    pad_value_variant = pad_value;
                } else if (tensor.dtype() == ttnn::DataType::INT32) {
                    TT_FATAL(
                        pad_value >= static_cast<float>(std::numeric_limits<int32_t>::min()) &&
                            pad_value < static_cast<float>(std::numeric_limits<int32_t>::max()),
                        "Pad value must be in the range of INT32 type");
                    // static_cast safely truncates the float into a signed integer,
                    // while std::bit_cast reinterprets those exact bits as unsigned to cleanly handle negative
                    // wrap-arounds.
                    pad_value_variant = std::bit_cast<uint32_t>(static_cast<int32_t>(pad_value));
                } else {
                    TT_FATAL(
                        pad_value >= 0.0f && pad_value < static_cast<float>(std::numeric_limits<uint32_t>::max()),
                        "Pad value must be in the range of UINT32 type");
                    pad_value_variant = (uint32_t)pad_value;
                }
                tensor = ttnn::operations::experimental::quasar::tilize_with_val_padding(
                    tensor,
                    Shape(padded_output_shape),
                    pad_value_variant,
                    output_memory_config,
                    dtype,
                    use_multicore_tilize,
                    sub_core_grids,
                    effective_tile);
            }
            if (original_rank < 2) {
                return ttnn::operations::experimental::quasar::reshape(
                    tensor,
                    original_shape,
                    std::nullopt /*Memory Config*/,
                    std::nullopt /*pad value*/,
                    TileReshapeMapMode::CACHE,
                    sub_core_grids);
            }
            return tensor;
        }
        TT_THROW("ttnn::to_layout: Unsupported output layout: {}!", layout);
    }
    TT_ASSERT(!dtype.has_value(), "dtype cannot be specified when converting layout on host!");
    if (!requires_padding_change(tensor, layout, effective_tile)) {
        return tensor.to_layout(
            layout, layout == ttnn::TILE_LAYOUT ? std::make_optional(effective_tile) : std::nullopt);
    }
    if (layout == ttnn::ROW_MAJOR_LAYOUT) {
        const auto source_tile = tensor.tensor_spec().tile();
        tensor = tensor.to_layout(layout);
        TT_FATAL(
            tensor.tensor_spec().tile() == source_tile,
            "ttnn::experimental::quasar::to_layout: host untilize must preserve source tile metadata for unpadding");
        tensor = tensor.unpad_from_tile(tensor.logical_shape());
        return ttnn::operations::experimental::quasar::reshape(
            tensor,
            ttnn::Shape{output_shape},
            std::nullopt, /*Memory Config*/
            std::nullopt, /*Pad Value*/
            TileReshapeMapMode::CACHE,
            sub_core_grids);
    }
    if (layout == ttnn::TILE_LAYOUT) {
        if (tensor.layout() == Layout::ROW_MAJOR && tensor.tensor_spec().tile() != effective_tile) {
            tensor = Tensor(tt::tt_metal::HostTensor::from_buffer(
                tensor.host_tensor().buffer(),
                TensorSpec(
                    tensor.logical_shape(),
                    tt::tt_metal::TensorLayout::fromPaddedShape(
                        tensor.dtype(),
                        tt::tt_metal::PageConfig(Layout::ROW_MAJOR, effective_tile),
                        tensor.memory_config(),
                        tensor.logical_shape(),
                        tensor.padded_shape())),
                tensor.tensor_topology()));
        }
        ttsl::SmallVector<uint32_t> padded_input_start;
        for (int index = 0; index < padded_output_shape.rank(); ++index) {
            padded_input_start.push_back(0);
        }
        tensor = tensor.pad(ttnn::Shape(padded_output_shape), ttnn::Shape(std::move(padded_input_start)), pad_value);
        tensor = tensor.to_layout(layout, effective_tile);
        return ttnn::experimental::view(tensor, output_shape, padded_output_shape);
    }
    TT_THROW("ttnn::to_layout: Unsupported output layout: {}!", layout);
}
}  // namespace

}  // namespace ttnn::operations::experimental::quasar::CMAKE_UNIQUE_NAMESPACE

namespace ttnn::operations::experimental::quasar {

Tensor to_layout(
    const Tensor& tensor_arg,
    Layout layout,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const float pad_value,
    const std::optional<tt::tt_metal::Tile>& tile) {
    return CMAKE_UNIQUE_NAMESPACE::to_layout_impl(
        tensor_arg, layout, dtype, memory_config, sub_core_grids, pad_value, tile);
}

}  // namespace ttnn::operations::experimental::quasar
