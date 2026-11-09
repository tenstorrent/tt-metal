// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_new/reshape_new.hpp"

#include <bit>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/tensor_layout_apis_with_custom_alignment.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_new/device/reshape_new_device_operation.hpp"
#include "ttnn/operations/data_movement/reshape_new/reshape_new_plan.hpp"
#include "ttnn/operations/data_movement/reshape_new/reshape_new_topology.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::data_movement::reshape_new {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using tt::tt_metal::BufferType;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::ShardOrientation;
using tt::tt_metal::TensorLayout;
using tt::tt_metal::TensorMemoryLayout;
using tt::tt_metal::TensorSpec;

uint32_t dim_from_end(const ttnn::Shape& shape, uint32_t from_end) {
    return shape.rank() >= from_end ? shape[-static_cast<int>(from_end)] : 1;
}

// Everything a reshape failure report needs (#29705, #33722).
std::string describe(
    const Tensor& input,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const std::optional<MemoryConfig>& memory_config) {
    return fmt::format(
        "input spec: {}, requested shape: {} (padded {}), requested memory config: {}",
        input.tensor_spec(),
        logical_shape,
        padded_shape,
        memory_config.has_value() ? fmt::format("{}", *memory_config) : std::string("<input's>"));
}

// Resolves a -1 dim from the input volume. A resolved request has no separate padded shape.
std::pair<ttnn::Shape, ttnn::Shape> resolve_inferred_dim(
    const Tensor& input, const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape) {
    std::optional<uint32_t> inferred;
    uint64_t known_volume = 1;
    for (uint32_t i = 0; i < logical_shape.rank(); ++i) {
        if (static_cast<int32_t>(logical_shape[i]) == -1) {
            TT_FATAL(
                !inferred.has_value(),
                "reshape: only one dimension can be inferred. {}",
                describe(input, logical_shape, padded_shape, std::nullopt));
            inferred = i;
        } else {
            known_volume *= logical_shape[i];
        }
    }
    if (!inferred.has_value()) {
        return {logical_shape, padded_shape};
    }
    const uint64_t input_volume = input.logical_volume();
    TT_FATAL(
        known_volume == 0 ? input_volume == 0 : input_volume % known_volume == 0,
        "reshape: cannot infer a dimension: {} elements do not split into the known dims. {}",
        input_volume,
        describe(input, logical_shape, padded_shape, std::nullopt));
    ttnn::Shape resolved = logical_shape;
    resolved[*inferred] = known_volume == 0 ? 0 : static_cast<uint32_t>(input_volume / known_volume);
    return {resolved, resolved};
}

// A padded shape the caller asked for is honored when it only pads dims -2 and -1 and, for TILE, is tile
// aligned there. Anything else is ignored in favor of the layout's own padding, as legacy does for TILE.
bool is_usable_padded_shape(const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape, const Tensor& input) {
    if (padded_shape == logical_shape) {
        return false;
    }
    if (padded_shape.rank() != logical_shape.rank() || logical_shape.rank() == 0) {
        return false;
    }
    const int rank = static_cast<int>(logical_shape.rank());
    for (int i = 0; i < rank; ++i) {
        if (padded_shape[i] < logical_shape[i] || (i < rank - 2 && padded_shape[i] != logical_shape[i])) {
            return false;
        }
    }
    if (input.layout() == Layout::TILE) {
        const auto tile = input.tensor_spec().tile().get_tile_shape();
        if (padded_shape[-1] % tile[1] != 0 || (rank >= 2 && padded_shape[-2] % tile[0] != 0)) {
            return false;
        }
    }
    return true;
}

TensorLayout make_layout(
    const Tensor& input,
    const ttnn::Shape& logical_shape,
    const std::optional<ttnn::Shape>& padded_shape,
    const MemoryConfig& memory_config) {
    const auto page_config = input.tensor_spec().page_config();
    if (padded_shape.has_value()) {
        return tt::tt_metal::experimental::tensor_layout_from_padded_shape(
            input.dtype(), page_config, memory_config, logical_shape, *padded_shape);
    }
    return TensorLayout(input.dtype(), page_config, memory_config);
}

TensorSpec make_spec(
    const Tensor& input,
    const ttnn::Shape& logical_shape,
    const std::optional<ttnn::Shape>& padded_shape,
    const MemoryConfig& memory_config) {
    return TensorSpec(logical_shape, make_layout(input, logical_shape, padded_shape, memory_config));
}

// The padded shape under which the output keeps the input's page sequence: the same padded last dim, and the
// same row padding (or none on either side). std::nullopt when no such shape exists.
std::optional<ttnn::Shape> view_padded_shape(const Tensor& input, const ttnn::Shape& logical_shape) {
    const auto& in_logical = input.logical_shape();
    const auto& in_padded = input.padded_shape();
    if (logical_shape.rank() < 2 || dim_from_end(logical_shape, 1) != dim_from_end(in_logical, 1)) {
        return std::nullopt;
    }
    const uint32_t height = dim_from_end(logical_shape, 2);
    const uint32_t in_height = dim_from_end(in_logical, 2);
    const uint32_t in_padded_height = dim_from_end(in_padded, 2);
    uint32_t padded_height = 0;
    if (height == in_height) {
        padded_height = in_padded_height;
    } else if (in_padded_height == in_height) {
        if (input.layout() == Layout::TILE && height % input.tensor_spec().tile().get_tile_shape()[0] != 0) {
            return std::nullopt;
        }
        padded_height = height;
    } else {
        return std::nullopt;
    }
    ttnn::Shape padded = logical_shape;
    padded[-1] = dim_from_end(in_padded, 1);
    padded[-2] = padded_height;
    return padded;
}

// A spec that keeps the input's buffer and distribution, if the output can be a view of the input under the
// input's own memory config.
std::optional<TensorSpec> view_spec(const Tensor& input, const ttnn::Shape& logical_shape) {
    const auto& in_spec = input.tensor_spec();
    const bool nd_sharded = is_nd_sharded_memory_config(input.memory_config());
    // An ND spec normalized to 2D was sized for the input's rank; the view keeps only the 2D spec.
    const MemoryConfig in_config = input.memory_config().created_with_nd_shard_spec()
                                       ? drop_normalized_nd_shard_spec(input.memory_config())
                                       : input.memory_config();
    if (const auto padded = view_padded_shape(input, logical_shape);
        padded.has_value() && (!nd_sharded || *padded == input.padded_shape())) {
        const auto spec = make_spec(input, logical_shape, padded, in_config);
        if (plan_reshape(in_spec, spec) == ReshapeKind::View) {
            return spec;
        }
    }
    // Shapes view_padded_shape does not cover (rank 0 and 1): the layout's own padding.
    if (!nd_sharded && tt::tt_metal::can_shape_fits_shard_grid(
                           make_layout(input, logical_shape, std::nullopt, in_config), logical_shape)) {
        const auto spec = make_spec(input, logical_shape, std::nullopt, in_config);
        if (plan_reshape(in_spec, spec) == ReshapeKind::View) {
            return spec;
        }
    }
    return std::nullopt;
}

// Largest n in [1, max_n] with dim % n == 0 and (dim / n) % align == 0, else 0.
uint32_t find_best_n_1d(uint32_t dim, uint32_t max_n, uint32_t align) {
    for (uint32_t n = max_n; n > 0; n--) {
        if (dim % n == 0 && (dim / n) % align == 0) {
            return n;
        }
    }
    return 0;
}

// Legacy's 2D shard-spec derivation (recompute_shard_spec_for_output), so default output specs match legacy:
// BLOCK keeps the grid origin and picks the largest evenly dividing sub-grid, HEIGHT/WIDTH keep the grid. A
// layout-only request is seeded from `input_shard_spec`. Falls back to INTERLEAVED when no grid fits.
MemoryConfig derive_2d_sharded_config(
    const MemoryConfig& memory_config,
    const TensorSpec& output_spec,
    bool explicit_memory_config,
    const std::optional<tt::tt_metal::ShardSpec>& input_shard_spec) {
    if (!memory_config.shard_spec().has_value() && input_shard_spec.has_value()) {
        return derive_2d_sharded_config(
            MemoryConfig{memory_config.memory_layout(), memory_config.buffer_type(), input_shard_spec},
            output_spec,
            /*explicit_memory_config=*/false,
            std::nullopt);
    }
    TT_FATAL(
        memory_config.shard_spec().has_value(),
        "reshape: sharded output memory_config (layout={}) was supplied without a shard_spec, and the input has "
        "none to derive one from. Provide a shard_spec or omit memory_config to inherit the input's.",
        memory_config.memory_layout());
    const auto& source_shard_spec = *memory_config.shard_spec();
    const auto orientation = source_shard_spec.orientation;
    const auto alignment = output_spec.page_config().get_recommended_shard_shape_alignment(output_spec.data_type());
    const uint32_t align_h = alignment.size() >= 2 ? alignment[-2] : 1;
    const uint32_t align_w = alignment.size() >= 1 ? alignment[-1] : 1;
    const uint32_t phys_h = output_spec.physical_shape().height();
    const uint32_t phys_w = output_spec.physical_shape().width();

    const auto bbox = source_shard_spec.grid.bounding_box();
    if (bbox.size() != source_shard_spec.grid.num_cores()) {
        log_warning(
            tt::LogOp,
            "reshape: shard grid is not rectangular (bounding box {} cores, grid {} cores); the output is "
            "INTERLEAVED",
            bbox.size(),
            source_shard_spec.grid.num_cores());
        return MemoryConfig{TensorMemoryLayout::INTERLEAVED, memory_config.buffer_type()};
    }
    const auto start = bbox.start_coord;
    const uint32_t grid_x = bbox.grid_size().x;
    const uint32_t grid_y = bbox.grid_size().y;
    const bool row_major = orientation == ShardOrientation::ROW_MAJOR;
    const auto layout = memory_config.memory_layout();

    if (layout == TensorMemoryLayout::BLOCK_SHARDED) {
        if (explicit_memory_config) {
            const uint32_t shard_h = source_shard_spec.shape[0];
            const uint32_t shard_w = source_shard_spec.shape[1];
            const uint32_t cores_h = row_major ? grid_y : grid_x;
            const uint32_t cores_w = row_major ? grid_x : grid_y;
            const bool aligned = (align_h == 0 || shard_h % align_h == 0) && (align_w == 0 || shard_w % align_w == 0);
            const bool covers = static_cast<uint64_t>(cores_h) * shard_h >= phys_h &&
                                static_cast<uint64_t>(cores_w) * shard_w >= phys_w;
            if (aligned && covers) {
                return memory_config;
            }
        }
        const uint32_t ny = find_best_n_1d(phys_h, row_major ? grid_y : grid_x, align_h);
        const uint32_t nx = find_best_n_1d(phys_w, row_major ? grid_x : grid_y, align_w);
        if (ny == 0 || nx == 0) {
            log_warning(
                tt::LogOp,
                "reshape: no BLOCK_SHARDED grid divides the output (physical {}x{}, alignment {}x{}); the output "
                "is INTERLEAVED",
                phys_h,
                phys_w,
                align_h,
                align_w);
            return MemoryConfig{TensorMemoryLayout::INTERLEAVED, memory_config.buffer_type()};
        }
        const CoreCoord end =
            row_major ? CoreCoord{start.x + nx - 1, start.y + ny - 1} : CoreCoord{start.x + ny - 1, start.y + nx - 1};
        return output_spec.block_sharded(CoreRange{start, end}, orientation).memory_config();
    }
    TT_FATAL(
        layout == TensorMemoryLayout::HEIGHT_SHARDED || layout == TensorMemoryLayout::WIDTH_SHARDED,
        "reshape: unsupported memory layout {}",
        layout);
    return layout == TensorMemoryLayout::HEIGHT_SHARDED
               ? output_spec.height_sharded(source_shard_spec.grid, orientation).memory_config()
               : output_spec.width_sharded(source_shard_spec.grid, orientation).memory_config();
}

// The output memory config when the output cannot be a view. An explicit config is used as given when it can
// hold the output; otherwise (and for an inherited config) the shard spec is re-derived for the output shape the
// way legacy does.
MemoryConfig resolve_output_memory_config(
    const Tensor& input,
    const ttnn::Shape& logical_shape,
    const std::optional<ttnn::Shape>& padded_shape,
    const MemoryConfig& memory_config,
    bool explicit_memory_config) {
    if (!memory_config.is_sharded()) {
        return memory_config;
    }
    const auto interleaved_spec = make_spec(
        input, logical_shape, padded_shape, MemoryConfig{TensorMemoryLayout::INTERLEAVED, memory_config.buffer_type()});
    if (is_nd_sharded_memory_config(memory_config)) {
        if (explicit_memory_config) {
            return memory_config;
        }
        return derive_nd_shard_spec_for_reshaped_output(
            memory_config, input.padded_shape(), interleaved_spec.padded_shape(), input.layout() == Layout::TILE);
    }
    if (explicit_memory_config && memory_config.shard_spec().has_value() &&
        tt::tt_metal::can_shape_fits_shard_grid(
            make_layout(input, logical_shape, padded_shape, memory_config), logical_shape)) {
        return memory_config;
    }
    return derive_2d_sharded_config(
        memory_config, interleaved_spec, explicit_memory_config, input.memory_config().shard_spec());
}

// Bits for one element of the pad value, repeated to fill 32 bits.
uint32_t pad_bits_for(DataType dtype, const PadValue& pad_value) {
    const auto as_float = [&]() -> float {
        return std::visit(
            [dtype](auto value) -> float {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, uint32_t>) {
                    if (dtype == DataType::FLOAT32) {
                        return std::bit_cast<float>(value);
                    }
                }
                return static_cast<float>(value);
            },
            pad_value);
    };
    const auto as_integer = [&]() -> uint32_t {
        return std::visit(
            [](auto value) -> uint32_t {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, uint32_t>) {
                    return value;
                } else {
                    return static_cast<uint32_t>(static_cast<int64_t>(value));
                }
            },
            pad_value);
    };
    switch (dtype) {
        case DataType::BFLOAT16:
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: return 0x00010001u * ::fp32_to_bf16_bits_round_to_nearest_even(as_float());
        case DataType::FLOAT32: return std::bit_cast<uint32_t>(as_float());
        case DataType::INT32:
        case DataType::UINT32: return as_integer();
        case DataType::UINT16: return 0x00010001u * (as_integer() & 0xFFFFu);
        case DataType::UINT8:
        case DataType::INT8: return 0x01010101u * (as_integer() & 0xFFu);
        default: return 0;
    }
}

// Applies the #64322 placement relabeling to a view or host result.
Tensor relabel(Tensor output, const Tensor& input) {
    const auto& topology = input.tensor_topology();
    auto remapped = remap_topology(topology, input.logical_shape(), output.logical_shape());
    if (remapped != output.tensor_topology()) {
        output.update_tensor_topology(remapped);
    }
    return output;
}

// Block-float data cannot be split element-wise: Gather runs on a BFLOAT16 copy with its padding filled (the shared
// exponents of the final typecast cover it), on interleaved intermediates, as legacy does, so the output bits match.
Tensor gather_block_float(
    const Tensor& input,
    const TensorSpec& output_spec,
    uint32_t pad_bits,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    const auto& target_config = output_spec.memory_config();
    Tensor working = input;
    if (input.memory_config().is_sharded()) {
        working = ttnn::to_memory_config(
            working, MemoryConfig{TensorMemoryLayout::INTERLEAVED, input.memory_config().buffer_type()});
    }
    working = ttnn::typecast(working, DataType::BFLOAT16, std::nullopt, std::nullopt, sub_core_grid);
    const MemoryConfig interleaved{TensorMemoryLayout::INTERLEAVED, target_config.buffer_type()};
    const auto bf16_spec = TensorSpec(
        output_spec.logical_shape(),
        tt::tt_metal::experimental::tensor_layout_from_padded_shape(
            DataType::BFLOAT16,
            output_spec.page_config(),
            interleaved,
            output_spec.logical_shape(),
            output_spec.padded_shape()));
    Tensor reshaped = ttnn::prim::reshape_new(
        working, bf16_spec, ReshapeKind::Gather, pad_bits, /*fill_padding=*/true, sub_core_grid);
    reshaped = ttnn::typecast(reshaped, input.dtype(), std::nullopt, std::nullopt, sub_core_grid);
    if (target_config.is_sharded()) {
        reshaped = ttnn::to_memory_config(reshaped, target_config);
    }
    return reshaped;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

Tensor reshape(
    const Tensor& input,
    const ttnn::Shape& requested_logical_shape,
    const ttnn::Shape& requested_padded_shape,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<PadValue>& pad_value,
    const std::optional<CoreRangeSet>& sub_core_grid,
    bool skip_padding_fill) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    const auto [logical_shape, padded_request] =
        resolve_inferred_dim(input, requested_logical_shape, requested_padded_shape);

    // Zero-volume tensors and host tensors are metadata only.
    if (input.logical_volume() == 0) {
        TT_FATAL(
            logical_shape.volume() == 0,
            "reshape: the input has no elements but the requested shape does. {}",
            describe(input, logical_shape, padded_request, memory_config));
        return relabel(ttnn::experimental::view(input, logical_shape, padded_request), input);
    }
    TT_FATAL(
        logical_shape.volume() != 0,
        "reshape: the requested shape has no elements but the input does. {}",
        describe(input, logical_shape, padded_request, memory_config));
    if (!is_device_tensor(input)) {
        return relabel(ttnn::experimental::view(input, logical_shape, padded_request), input);
    }

    if (logical_shape.volume() != input.logical_volume()) {
        // Kept from legacy (#15137, #15558): a TILE tensor may be relabeled to a different logical volume within
        // the same tile-aligned padded shape, exposing or hiding padding.
        const auto tile = input.tensor_spec().tile().get_tile_shape();
        const bool tile_view = input.layout() == Layout::TILE && padded_request.rank() >= 2 &&
                               padded_request[-2] % tile[0] == 0 && padded_request[-1] % tile[1] == 0 &&
                               input.padded_shape()[-1] == padded_request[-1];
        TT_FATAL(
            tile_view,
            "reshape: cannot reshape {} elements into {} elements. {}",
            input.logical_volume(),
            logical_shape.volume(),
            describe(input, logical_shape, padded_request, memory_config));
        return ttnn::experimental::view(input, logical_shape, padded_request);
    }

    const MemoryConfig requested_config = memory_config.value_or(input.memory_config());
    const bool explicit_config = memory_config.has_value();
    const bool same_config = is_functionally_same_memory_config(requested_config, input.memory_config());
    const bool padding_unchanged = padded_request == logical_shape || padded_request == input.padded_shape();

    // Same shape: nothing to do, or a pure memory-config conversion. A layout-only sharded request (no shard spec)
    // goes through derivation below.
    if (logical_shape == input.logical_shape() && padding_unchanged) {
        if (same_config) {
            return input;
        }
        const bool layout_only = requested_config.is_sharded() && !requested_config.shard_spec().has_value() &&
                                 !requested_config.nd_shard_spec().has_value();
        if (!layout_only) {
            return ttnn::to_memory_config(input, requested_config);
        }
    }

    const std::optional<ttnn::Shape> padded_shape =
        is_usable_padded_shape(logical_shape, padded_request, input) ? std::optional(padded_request) : std::nullopt;

    // A view when the input's own buffer can be relabeled.
    if (same_config && !padded_shape.has_value()) {
        if (const auto spec = view_spec(input, logical_shape); spec.has_value()) {
            return relabel(ttnn::experimental::view(input, spec->logical_shape(), spec->padded_shape()), input);
        }
    }

    // A config sized for the old shape cannot describe the new one: drop a normalized ND spec before deriving.
    const MemoryConfig output_config = resolve_output_memory_config(
        input, logical_shape, padded_shape, drop_normalized_nd_shard_spec(requested_config), explicit_config);
    const TensorSpec output_spec = make_spec(input, logical_shape, padded_shape, output_config);

    switch (plan_reshape(input.tensor_spec(), output_spec)) {
        case ReshapeKind::NoOp: return input;
        case ReshapeKind::View:
            return relabel(
                ttnn::experimental::view(input, output_spec.logical_shape(), output_spec.padded_shape()), input);
        case ReshapeKind::PageCopy:
            return ttnn::prim::reshape_new(
                input, output_spec, ReshapeKind::PageCopy, /*pad_bits=*/0, /*fill_padding=*/false, sub_core_grid);
        case ReshapeKind::Gather: {
            const uint32_t pad_bits = pad_bits_for(input.dtype(), pad_value.value_or(PadValue{0.0f}));
            if (is_block_float(input.dtype())) {
                return gather_block_float(input, output_spec, pad_bits, sub_core_grid);
            }
            const auto& physical = output_spec.physical_shape();
            const bool has_padding =
                static_cast<uint64_t>(physical.height()) * physical.width() != output_spec.logical_shape().volume();
            return ttnn::prim::reshape_new(
                input, output_spec, ReshapeKind::Gather, pad_bits, has_padding && !skip_padding_fill, sub_core_grid);
        }
    }
    TT_THROW("reshape: unhandled plan. {}", describe(input, logical_shape, padded_request, memory_config));
}

}  // namespace ttnn::operations::data_movement::reshape_new
