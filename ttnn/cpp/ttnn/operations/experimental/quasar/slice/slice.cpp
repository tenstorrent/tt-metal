// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/slice/slice.hpp"
#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/common/synthesize_output_shard_spec.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/data_movement/transpose/device/transpose_utils.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/experimental/quasar/to_layout/to_layout_op.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/range_lockstep_allocation/memory_config.hpp>

namespace ttnn::operations::experimental::quasar {

namespace detail {

inline bool is_rm_bw_sharded(const tt::tt_metal::MemoryConfig& mc) {
    const auto layout = mc.memory_layout();
    return mc.is_sharded() && (layout == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED ||
                               layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED);
}

// True ND sharding gives an RM page one shard row, so like B/W sharding a tensor row can span several pages.
inline bool is_nd_sharded(const tt::tt_metal::MemoryConfig& mc) {
    return mc.memory_layout() == tt::tt_metal::TensorMemoryLayout::ND_SHARDED && mc.nd_shard_spec().has_value();
}

// noc_async_*_sharded takes the number of pages an RM row spans from the buffer's distribution spec, whose
// ranks the allocator squeezes. When squeezing folds the row's pages together with other dims (e.g. shard
// [1, 1, 1, 8] on a 64-wide tensor collapses to rank one), the helper sees the wrong count and strides rows
// by the wrong number of pages.
inline bool has_squeezed_rm_row_pages(const tt::tt_metal::TensorSpec& spec) {
    const auto& mc = spec.memory_config();
    if (spec.layout() != Layout::ROW_MAJOR || !is_nd_sharded(mc) || spec.padded_shape().rank() < 2 ||
        spec.logical_shape().volume() == 0) {
        return false;
    }
    const auto distribution = spec.compute_buffer_sharding_args().buffer_distribution_spec();
    if (!distribution.has_value()) {
        return false;
    }
    const auto& tensor_pages = distribution->tensor_shape_in_pages();
    const auto& shard_pages = distribution->shard_shape_in_pages();
    const uint32_t seen_pages_per_row = tensor_pages.rank() > 1 ? tt::div_up(tensor_pages[-1], shard_pages[-1]) : 1;
    const uint32_t pages_per_row = tt::div_up(spec.padded_shape()[-1], mc.nd_shard_spec()->shard_shape[-1]);
    return seen_pages_per_row != pages_per_row;
}

// The composite fallback converts its result into the requested ND config without running the device op, so
// it repeats the device op's CONTIGUOUS_1D shard-count check on the spec it is about to create.
inline void validate_contiguous_nd_output(
    const tt::tt_metal::MemoryConfig& mc,
    const ttnn::Shape& output_shape,
    DataType dtype,
    const PageConfig& page_config) {
    if (!is_nd_sharded(mc)) {
        return;
    }
    ttnn::operations::experimental::quasar::validate_contiguous_nd_output_spec(
        tt::tt_metal::TensorSpec(output_shape, tt::tt_metal::TensorLayout(dtype, page_config, mc)));
}

// Rescale an inherited true-ND config to the sliced output, keeping its grid, distribution strategy and
// range-lockstep flag. Any other config is returned unchanged.
inline tt::tt_metal::MemoryConfig rescale_inherited_nd_config(
    const tt::tt_metal::MemoryConfig& mc,
    const ttnn::Tensor& input,
    const ttnn::Shape& output_padded_shape,
    Layout output_layout) {
    if (!is_nd_sharded(mc)) {
        return mc;
    }
    const auto& nd_shard_spec = mc.nd_shard_spec().value();
    const auto& input_padded_shape = input.padded_shape();
    if (nd_shard_spec.shard_shape.rank() > output_padded_shape.rank() ||
        input_padded_shape.rank() != output_padded_shape.rank()) {
        return mc;
    }
    auto new_nd_spec = ttnn::operations::data_movement::common::rescale_nd_shard_spec_for_output(
        nd_shard_spec,
        input_padded_shape,
        output_padded_shape,
        // A TILE input is untilized into the RM staging tensor, which can't keep a block-float dtype.
        output_layout == Layout::ROW_MAJOR && input.layout() == Layout::TILE
            ? ttnn::operations::data_movement::untilize_output_dtype(input.dtype())
            : input.dtype(),
        PageConfig(output_layout, input.tensor_spec().tile()),
        mc.buffer_type());
    if (new_nd_spec == nd_shard_spec) {
        return mc;
    }
    auto rescaled = tt::tt_metal::MemoryConfig(mc.buffer_type(), new_nd_spec);
    tt::tt_metal::experimental::range_lockstep_allocation::set_range_lockstep_allocation(
        rescaled, tt::tt_metal::experimental::range_lockstep_allocation::is_range_lockstep_allocation(mc));
    return rescaled;
}

// RM constraint, not TILE: `noc_async_*_sharded` derives pages_per_row from the padded shape,
// so non-tile-multiple H/W overshoots.
inline bool has_nontile_hw(const ttnn::Tensor& input) {
    const auto& s = input.logical_shape();
    if (s.rank() < 2) {
        return false;
    }
    return s[-1] % tt::constants::TILE_WIDTH != 0 || s[-2] % tt::constants::TILE_HEIGHT != 0;
}

// For B/W-sharded buffers the NOC helper splits by W only; irregular H alone is fine natively.
inline bool has_nontile_w(const ttnn::Tensor& input) {
    const auto& s = input.logical_shape();
    return s.rank() >= 1 && s[-1] % tt::constants::TILE_WIDTH != 0;
}

// has_subaligned_shard_row covers only legacy B/W specs; a true ND spec's shard row is its last dim.
inline bool has_subaligned_shard_row(const tt::tt_metal::MemoryConfig& mc, uint32_t element_size) {
    if (ttnn::operations::data_movement::has_subaligned_shard_row(mc, element_size)) {
        return true;
    }
    if (!is_nd_sharded(mc)) {
        return false;
    }
    const uint32_t alignment = mc.buffer_type() == tt::tt_metal::BufferType::DRAM
                                   ? tt::tt_metal::hal::get_dram_alignment()
                                   : tt::tt_metal::hal::get_l1_alignment();
    return (mc.nd_shard_spec()->shard_shape[-1] * element_size) % alignment != 0;
}

// Geometry of a TILE tensor once rm_only untilizes it into `staging_mc`: the dtype loses its block-float
// form, and the page becomes one shard row.
inline tt::tt_metal::TensorSpec rm_staging_spec(
    const ttnn::Tensor& tile_input, const tt::tt_metal::MemoryConfig& staging_mc) {
    return tt::tt_metal::TensorSpec(
        tile_input.logical_shape(),
        tt::tt_metal::TensorLayout(
            ttnn::operations::data_movement::untilize_output_dtype(tile_input.dtype()),
            PageConfig(Layout::ROW_MAJOR),
            staging_mc));
}

// Route RM sharded input through composite when native isn't safe: nontile-aligned B/W, B/W or ND with
// a sub-aligned shard row or non-zero width-begin, ND whose row pages are squeezed away, or
// nontile-aligned HEIGHT/ND outside the sharded fast path.
// A TILE input that rm_only untilizes into `staging_mc` (staged_rm) is read as ROW_MAJOR, so the ND checks
// apply to that staging tensor rather than to the TILE one.
inline bool needs_rm_composite_input(
    const ttnn::Tensor& input,
    const tt::tt_metal::MemoryConfig& output_mc,
    bool no_step,
    bool width_begin_nonzero,
    bool staged_rm,
    const tt::tt_metal::MemoryConfig& staging_mc) {
    if (staged_rm && input.layout() == Layout::TILE) {
        if (!is_nd_sharded(staging_mc)) {
            return false;
        }
        const auto spec = rm_staging_spec(input, staging_mc);
        const uint32_t element_size = spec.data_type() == input.dtype() ? input.element_size() : 2;
        return width_begin_nonzero || has_subaligned_shard_row(staging_mc, element_size) ||
               has_squeezed_rm_row_pages(spec);
    }
    if (input.layout() != Layout::ROW_MAJOR || !input.is_sharded()) {
        return false;
    }
    if (is_rm_bw_sharded(input.memory_config())) {
        // Only W misalignment breaks the per-shard page split; irregular H is fine.
        return has_nontile_w(input) || width_begin_nonzero ||
               has_subaligned_shard_row(input.memory_config(), input.element_size());
    }
    // SliceRmProgramFactory applies the width-begin to the buffer base, which only lands inside the row
    // when the row is a single page.
    if (is_nd_sharded(input.memory_config()) &&
        (width_begin_nonzero || has_subaligned_shard_row(input.memory_config(), input.element_size()) ||
         has_squeezed_rm_row_pages(input.tensor_spec()))) {
        return true;
    }
    // Require a spec: no-spec HEIGHT output triggers needs_sharded_output_reshard → composite anyway.
    const bool stays_on_sharded_fast_path =
        output_mc.is_sharded() && output_mc.shard_spec().has_value() &&
        output_mc.memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED && no_step;
    return has_nontile_hw(input) && !stays_on_sharded_fast_path;
}

// Compose RM B/W- or ND-sharded output on nontile-aligned W, a sub-aligned shard row, or ND row pages
// squeezed away (irregular H is fine natively). Callers must pass the post-rescale config, since the
// implicit-inheritance rescale can turn an aligned inherited shard row into a sub-aligned one.
inline bool needs_rm_composite_output(
    const ttnn::Tensor& input,
    const tt::tt_metal::MemoryConfig& output_mc,
    const ttnn::Shape& output_shape,
    bool staged_rm) {
    // The staged output is ROW_MAJOR but the TILE input is not, so only the ND checks below apply to it.
    const bool tile_staged = staged_rm && input.layout() == Layout::TILE;
    if ((input.layout() != Layout::ROW_MAJOR && !tile_staged) ||
        !(is_rm_bw_sharded(output_mc) || is_nd_sharded(output_mc)) || (tile_staged && !is_nd_sharded(output_mc))) {
        return false;
    }
    const uint32_t element_size = tile_staged && input.dtype() == DataType::BFLOAT8_B ? 2 : input.element_size();
    if ((!tile_staged && has_nontile_w(input)) || has_subaligned_shard_row(output_mc, element_size)) {
        return true;
    }
    // A shard wider than what is left of the row pads the output row past the input's, and the kernels
    // copy whole padded rows, so the last read would run past the input row.
    if (is_nd_sharded(output_mc) && output_shape[-1] % output_mc.nd_shard_spec()->shard_shape[-1] != 0) {
        return true;
    }
    return is_nd_sharded(output_mc) && has_squeezed_rm_row_pages(tt::tt_metal::TensorSpec(
                                           output_shape,
                                           tt::tt_metal::TensorLayout(
                                               ttnn::operations::data_movement::untilize_output_dtype(input.dtype()),
                                               PageConfig(Layout::ROW_MAJOR),
                                               output_mc)));
}

// Sharded-no-spec output that can't seed from the input (not sharded, or layout differs);
// synthesize from the sliced shape.
inline bool needs_sharded_output_reshard(const ttnn::Tensor& input, const tt::tt_metal::MemoryConfig& output_mc) {
    if (!output_mc.is_sharded() || output_mc.shard_spec().has_value()) {
        return false;
    }
    if (!input.is_sharded()) {
        return true;
    }
    return input.memory_config().memory_layout() != output_mc.memory_layout();
}

}  // namespace detail

template <typename T>
ttnn::Tensor slice(
    const ttnn::Tensor& input_tensor,
    ttsl::Span<const T> begins,
    ttsl::Span<const T> ends,
    ttsl::Span<const T> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    // Ensure start and end vectors have matching sizes and correct tensor rank

    const auto& input_shape = input_tensor.logical_shape();
    uint32_t input_rank = input_shape.rank();
    auto input_layout = input_tensor.layout();

    if (input_rank == 0) {
        return input_tensor;
    }
    TT_FATAL(
        input_rank == begins.size(), "Input rank {} and begins {} must have the same size", input_rank, begins.size());
    TT_FATAL(begins.size() == ends.size(), "Start {} and end {} must have the same size", begins.size(), ends.size());
    TT_FATAL(
        step.size() == begins.size(),
        "Step {} must have the same size as start {} and end",
        step.size(),
        begins.size());
    TT_FATAL(std::ranges::all_of(step, [](T s) { return s > 0; }), "Slice step must be greater than 0 in every dim");

    bool no_step = std::ranges::all_of(step, [](uint32_t s) { return s == 1; });
    bool starts_zero = std::ranges::all_of(begins, [](uint32_t s) { return s == 0; });
    bool ends_max = true;
    for (size_t i = 0; i < ends.size(); ++i) {
        ends_max &= ends[i] == input_shape[i];
        if (!ends_max) {
            break;
        }
    }

    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();

    auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config()
                                                            : memory_config_arg.value_or(input_tensor.memory_config());
    // memory_config: original (input-compatible) config, used by to_layout in the rm_only path.
    // output_memory_config: may be rescaled below to match the sliced output dims.
    auto output_memory_config = memory_config;
    // Config the result is finally placed in. Differs from output_memory_config only when a TILE input is
    // sliced through ROW_MAJOR staging: the staging shards may be too short for tiles, so the restored TILE
    // result gets its own tile-compatible config.
    std::optional<MemoryConfig> final_output_memory_config;

    // Fill in a missing shard_spec on a sharded output: reuse source's if layouts match, else
    // synthesize from the source shape. A populated nd_shard_spec is already fully specified.
    auto resolve_mc = [&](const ttnn::Tensor& source) {
        auto resolved_mc = final_output_memory_config.value_or(output_memory_config);
        if (resolved_mc.is_sharded() && !resolved_mc.shard_spec().has_value() &&
            !resolved_mc.nd_shard_spec().has_value()) {
            const auto& in_mc = source.memory_config();
            if (in_mc.is_sharded() && in_mc.memory_layout() == resolved_mc.memory_layout() &&
                in_mc.shard_spec().has_value()) {
                resolved_mc =
                    ttnn::MemoryConfig(resolved_mc.memory_layout(), resolved_mc.buffer_type(), in_mc.shard_spec());
            } else {
                auto spec = operations::data_movement::transpose::generate_transpose_shard_spec(
                    source, source.padded_shape(), resolved_mc.memory_layout());
                resolved_mc = ttnn::MemoryConfig(resolved_mc.memory_layout(), resolved_mc.buffer_type(), spec);
            }
        }
        return resolved_mc;
    };

    // True when the next to_memory_config can land directly in the preallocated buffer: layouts
    // must match (prim::copy validates equality) and source must not alias the preallocated. A dtype
    // mismatch is left to finalize_into_preallocated to reject.
    auto can_land_in_preallocated = [&](const ttnn::Tensor& source) {
        return optional_output_tensor.has_value() && source.storage_type() == StorageType::DEVICE &&
               source.layout() == optional_output_tensor->layout() &&
               source.dtype() == optional_output_tensor->dtype() && source.buffer() != optional_output_tensor->buffer();
    };
    // Safety net for paths that bypass the device op (composite / no-op / rm_only layout fixup):
    // short-circuits by buffer identity, falls back to ttnn::copy when result != preallocated.
    auto finalize_into_preallocated = [&](const ttnn::Tensor& result) -> ttnn::Tensor {
        if (!optional_output_tensor.has_value() || result.storage_type() != StorageType::DEVICE) {
            return result;
        }
        const auto& dst = optional_output_tensor.value();
        if (result.buffer() == dst.buffer()) {
            return result;
        }
        // ttnn::copy converts silently, so only accept the result's dtype, or the input's: the rm_only
        // path promotes BFLOAT8_B to BFLOAT16 (untilize), and copying back only undoes that promotion.
        TT_FATAL(
            dst.dtype() == result.dtype() || dst.dtype() == input_tensor.dtype(),
            "The preallocated output tensor needs dtype {} or {}, however it is {}",
            result.dtype(),
            input_tensor.dtype(),
            dst.dtype());
        return ttnn::copy(result, dst);
    };

    auto ret_adjustment([&](const ttnn::Tensor& source) {
        if (source.storage_type() != StorageType::DEVICE) {
            return source;
        }
        // source carries the correct shard spec (input for no-op, prim::slice result otherwise).
        const auto resolved_mc = resolve_mc(source);
        const auto target = can_land_in_preallocated(source) ? optional_output_tensor : std::nullopt;
        auto tensor = ttnn::to_memory_config(source, resolved_mc, std::nullopt, target);
        tensor = ttnn::operations::experimental::quasar::to_layout(tensor, input_layout);
        return tensor;
    });

    // No-op check
    if (no_step && starts_zero && ends_max) {
        // The no-op converts the input into the requested config without the device op, so check it here.
        if (input_tensor.storage_type() == StorageType::DEVICE) {
            detail::validate_contiguous_nd_output(
                resolve_mc(input_tensor),
                input_shape,
                input_tensor.dtype(),
                input_tensor.tensor_spec().tensor_layout().get_page_config());
        }
        return finalize_into_preallocated(ret_adjustment(input_tensor));
    }

    // Create modified vectors with wrapped indices and adjust them to match the tensor's rank
    ttsl::SmallVector<uint32_t> modified_begins(input_rank, 0);
    ttsl::SmallVector<uint32_t> modified_ends(input_rank, 0);
    ttsl::SmallVector<uint32_t> modified_step(input_rank, 1);

    // Wrap indices and adjust begins, ends, and step
    for (size_t i = 0; i < begins.size(); ++i) {
        if constexpr (std::is_signed_v<T>) {
            modified_begins[i] = operations::data_movement::wrap_index(begins[i], input_shape[i]);
            modified_ends[i] = operations::data_movement::wrap_index(ends[i], input_shape[i]);
            modified_step[i] = static_cast<uint32_t>(step[i]);
        } else {
            modified_begins[i] = begins[i];
            modified_ends[i] = ends[i];
            modified_step[i] = step[i];
        }
    }

    auto output_dim_i = [&modified_begins, &modified_step](size_t i, const ttsl::SmallVector<uint32_t>& modified_ends) {
        return (modified_ends[i] - modified_begins[i] + modified_step[i] - 1) / modified_step[i];
    };

    auto check_handled_tile_alignment = [&modified_begins, &input_rank, &tile_shape]() -> bool {
        return (
            modified_begins[input_rank - 1] % tile_shape[1] == 0 &&
            modified_begins[input_rank - 2] % tile_shape[0] == 0);
    };

    bool rm_only = false;
    bool one_dimensional = input_rank == 1;
    bool handled_tile_alignment = one_dimensional ? true : check_handled_tile_alignment();

    // Use the RM path when input isn't TILE, or TILE input has strided/1D/non-tile-aligned begins
    // (ends are padded downstream, so only begin alignment matters).
    rm_only = (input_tensor.layout() != Layout::TILE) || (!no_step || one_dimensional || !handled_tile_alignment);

    // Implicit inheritance from a sharded input: rescale the shard spec to the sliced shape so the
    // output doesn't reuse the input's (oversized) spec. Covers HEIGHT/WIDTH/BLOCK. (Issue #38016)
    // Runs before the composite decision below so both that decision and the composite path's own
    // resolve_mc see the final spec — rescaling can turn an aligned shard row into a sub-aligned one.
    if (!memory_config_arg.has_value() && !optional_output_tensor.has_value() && input_tensor.is_sharded()) {
        const auto& mem_layout = output_memory_config.memory_layout();
        if (input_rank >= 2 && (mem_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED ||
                                mem_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED ||
                                mem_layout == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED)) {
            const auto& shard_spec_val = output_memory_config.shard_spec().value();

            // Compute output dimensions, tile-aligned if using TILE path
            ttsl::SmallVector<uint32_t> output_dims(input_rank);
            for (size_t i = 0; i < input_rank; i++) {
                output_dims[i] = output_dim_i(i, modified_ends);
            }
            if (!rm_only) {
                output_dims[input_rank - 2] =
                    std::max(tt::round_up(output_dims[input_rank - 2], tile_shape[0]), tile_shape[0]);
                output_dims[input_rank - 1] =
                    std::max(tt::round_up(output_dims[input_rank - 1], tile_shape[1]), tile_shape[1]);
            }

            // Flatten to 2D: height = product of all dims except last, width = last dim
            uint32_t output_height = 1;
            for (size_t i = 0; i + 1 < input_rank; i++) {
                output_height *= output_dims[i];
            }
            uint32_t output_width = output_dims[input_rank - 1];

            std::array<uint32_t, 2> new_shard_shape = shard_spec_val.shape;
            if (mem_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
                uint32_t num_cores = shard_spec_val.num_cores();
                uint32_t new_shard_h = tt::div_up(output_height, num_cores);
                if (!rm_only) {
                    new_shard_h = std::max(tt::round_up(new_shard_h, tile_shape[0]), tile_shape[0]);
                }
                new_shard_shape = {new_shard_h, output_width};
            } else if (mem_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
                uint32_t num_cores = shard_spec_val.num_cores();
                uint32_t new_shard_w = tt::div_up(output_width, num_cores);
                if (!rm_only) {
                    new_shard_w = std::max(tt::round_up(new_shard_w, tile_shape[1]), tile_shape[1]);
                }
                new_shard_shape = {output_height, new_shard_w};
            } else {
                // BLOCK_SHARDED requires a rectangular grid; bounding_box() is valid only then.
                const auto bbox = shard_spec_val.grid.bounding_box();
                const uint32_t grid_h = bbox.end_coord.y - bbox.start_coord.y + 1;
                const uint32_t grid_w = bbox.end_coord.x - bbox.start_coord.x + 1;
                TT_FATAL(
                    shard_spec_val.num_cores() == grid_h * grid_w,
                    "BLOCK_SHARDED grid must be a full rectangle; got {} cores for {}x{} bounding box",
                    shard_spec_val.num_cores(),
                    grid_h,
                    grid_w);
                uint32_t new_shard_h = tt::div_up(output_height, grid_h);
                uint32_t new_shard_w = tt::div_up(output_width, grid_w);
                if (!rm_only) {
                    new_shard_h = std::max(tt::round_up(new_shard_h, tile_shape[0]), tile_shape[0]);
                    new_shard_w = std::max(tt::round_up(new_shard_w, tile_shape[1]), tile_shape[1]);
                }
                new_shard_shape = {new_shard_h, new_shard_w};
            }

            if (new_shard_shape != shard_spec_val.shape) {
                auto new_shard_spec =
                    tt::tt_metal::ShardSpec(shard_spec_val.grid, new_shard_shape, shard_spec_val.orientation);
                output_memory_config = MemoryConfig(
                    output_memory_config.memory_layout(), output_memory_config.buffer_type(), new_shard_spec);
            }
        } else if (detail::is_nd_sharded(output_memory_config)) {
            // True ND sharding (e.g. CONTIGUOUS_1D) has no legacy 2D equivalent, so the HEIGHT/WIDTH/
            // BLOCK rescale above can't apply; rescale the ND shard shape to the sliced output dims
            // instead, preserving the caller's grid/distribution strategy.
            ttsl::SmallVector<uint32_t> output_dims(input_rank);
            for (size_t i = 0; i < input_rank; i++) {
                output_dims[i] = output_dim_i(i, modified_ends);
            }
            if (!rm_only) {
                output_dims[input_rank - 2] =
                    std::max(tt::round_up(output_dims[input_rank - 2], tile_shape[0]), tile_shape[0]);
                output_dims[input_rank - 1] =
                    std::max(tt::round_up(output_dims[input_rank - 1], tile_shape[1]), tile_shape[1]);
            }
            const auto staging_dims = ttnn::Shape(output_dims);
            output_memory_config = detail::rescale_inherited_nd_config(
                output_memory_config, input_tensor, staging_dims, rm_only ? Layout::ROW_MAJOR : Layout::TILE);
            if (rm_only && input_layout == Layout::TILE) {
                // A rank-1 tensor is tiled as a single row of tiles, so rescale against the promoted
                // [tile_h, width] geometry; the logical output rank is untouched.
                ttsl::SmallVector<uint32_t> tile_dims(output_dims);
                if (input_rank == 1) {
                    tile_dims = {tile_shape[0], output_dims[0]};
                }
                const size_t tile_rank = tile_dims.size();
                tile_dims[tile_rank - 2] =
                    std::max(tt::round_up(tile_dims[tile_rank - 2], tile_shape[0]), tile_shape[0]);
                tile_dims[tile_rank - 1] =
                    std::max(tt::round_up(tile_dims[tile_rank - 1], tile_shape[1]), tile_shape[1]);
                final_output_memory_config = detail::rescale_inherited_nd_config(
                    memory_config, input_tensor, ttnn::Shape(tile_dims), Layout::TILE);
            }
        }
    }

    // Composite hop: unshard to L1 interleaved if needed, slice, then convert to the requested mc.
    // Must stay below the rescale: needs_rm_composite_output and resolve_mc need the rescaled spec,
    // which can be sub-aligned even when the inherited one isn't.
    const bool width_begin_nonzero = !begins.empty() && begins.back() != 0;
    const bool rm_in_bad = detail::needs_rm_composite_input(
        input_tensor, output_memory_config, no_step, width_begin_nonzero, rm_only, memory_config);
    ttsl::SmallVector<uint32_t> sliced_dims(input_rank);
    for (size_t i = 0; i < input_rank; i++) {
        sliced_dims[i] = output_dim_i(i, modified_ends);
    }
    const bool rm_out_bad =
        detail::needs_rm_composite_output(input_tensor, output_memory_config, ttnn::Shape(sliced_dims), rm_only);
    const bool out_no_spec = detail::needs_sharded_output_reshard(input_tensor, output_memory_config);
    if (rm_in_bad || rm_out_bad || out_no_spec) {
        detail::validate_contiguous_nd_output(
            final_output_memory_config.value_or(output_memory_config),
            ttnn::Shape(sliced_dims),
            input_tensor.dtype(),
            input_tensor.tensor_spec().tensor_layout().get_page_config());
        const auto interleaved_l1 =
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1);
        Tensor x = rm_in_bad ? ttnn::to_memory_config(input_tensor, interleaved_l1, std::nullopt) : input_tensor;
        // Intermediate lives in L1 interleaved; to_memory_config lands the result in the caller's
        // buffer. sub_core_grids is threaded through to bound the recursive slice's work split.
        auto sliced = slice<T>(x, begins, ends, step, interleaved_l1, std::nullopt, pad_value, sub_core_grids);
        // slice preserves layout and to_memory_config doesn't change it — no trailing to_layout needed.
        // sliced is L1-interleaved so resolve_mc falls through to generate_transpose_shard_spec.
        const auto final_mc = resolve_mc(sliced);
        if (sliced.memory_config() == final_mc) {
            return finalize_into_preallocated(sliced);
        }
        const auto target = can_land_in_preallocated(sliced) ? optional_output_tensor : std::nullopt;
        return finalize_into_preallocated(ttnn::to_memory_config(sliced, final_mc, std::nullopt, target));
    }

    Tensor input = input_tensor;
    if (rm_only) {
        if (!no_step) {
            TT_FATAL(input.dtype() != DataType::BFLOAT8_B, "Strided slice is not supported for BFLOAT8 tensors");
        }
        input =
            ttnn::operations::experimental::quasar::to_layout(input, Layout::ROW_MAJOR, std::nullopt, memory_config);
    }

    ttsl::SmallVector<uint32_t> padded_ends = modified_ends;
    if (input.layout() == Layout::TILE) {
        padded_ends[input_rank - 2] = std::max(tt::round_up(padded_ends[input_rank - 2], tile_shape[0]), tile_shape[0]);
        padded_ends[input_rank - 1] = std::max(tt::round_up(padded_ends[input_rank - 1], tile_shape[1]), tile_shape[1]);
    }

    ttsl::SmallVector<uint32_t> actual_shape_vec, final_padded_shape_vec;
    actual_shape_vec.reserve(input_rank);
    final_padded_shape_vec.reserve(input_rank);
    bool empty = false;

    // Compute actual and padded shapes for the original input rank
    for (size_t i = 0; i < input_rank; ++i) {
        TT_FATAL(
            modified_ends[i] >= modified_begins[i],
            "End {} must be greater than or equal to start {}",
            modified_ends[i],
            modified_begins[i]);
        auto val = output_dim_i(i, modified_ends);
        if (val == 0) {
            empty = true;
        }
        actual_shape_vec.push_back(val);
        final_padded_shape_vec.push_back(std::max(output_dim_i(i, padded_ends), static_cast<uint32_t>(1)));
    }
    ttnn::Shape actual_shape(actual_shape_vec);
    ttnn::Shape final_padded_shape(final_padded_shape_vec);

    if (empty) {
        TT_FATAL(
            input.storage_type() == StorageType::DEVICE, "Host tensor slice cannot return a scalar or empty tensor");
        if (optional_output_tensor.has_value()) {
            return optional_output_tensor.value();
        }
        return ttnn::empty(
            actual_shape,
            input_tensor.dtype(),
            input_tensor.layout(),
            input_tensor.device(),
            memory_config_arg.value_or(input_tensor.memory_config()));
    }
    // prim::qsr::slice writes into the caller's buffer verbatim, so it can only be the destination when
    // the buffer's page geometry matches the output the op computes for `input`. rm_only downgrades a
    // TILE input to ROW_MAJOR, which leaves the caller's TILE tensor tile-paged while the RM factories
    // page it by row; hand it over only when the layouts agree and let ret_adjustment retilize into it
    // via finalize_into_preallocated's copy otherwise.
    const auto prim_output = can_land_in_preallocated(input) ? optional_output_tensor : std::nullopt;
    auto res = ttnn::prim::qsr::slice(
        input,
        ttnn::Shape(modified_begins),
        ttnn::Shape(padded_ends),
        ttnn::Shape(modified_step),
        output_memory_config,
        /*use_tensor_args*/ false,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        sub_core_grids,
        prim_output);
    res = ttnn::experimental::view(res, actual_shape, final_padded_shape);

    auto dim_needs_fill = [&input_shape, &actual_shape, &final_padded_shape](int i) {
        return ((actual_shape[i] != final_padded_shape[i]) && (input_shape[i] != actual_shape[i]));
    };

    if (pad_value.has_value() && (dim_needs_fill(-1) || dim_needs_fill(-2))) {
        res = ttnn::fill_implicit_tile_padding(res, pad_value.value());
    }

    // ret_adjustment may re-allocate (rm_only-from-TILE); finalize guarantees the caller's buffer.
    return finalize_into_preallocated(ret_adjustment(res));
}

template <typename T, std::size_t N>
ttnn::Tensor slice(
    const ttnn::Tensor& input_tensor,
    const std::array<T, N>& output_tensor_start,
    const std::array<T, N>& output_tensor_end,
    const std::array<T, N>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    ttsl::Span<const T> start(output_tensor_start.begin(), output_tensor_start.end());
    ttsl::Span<const T> end(output_tensor_end.begin(), output_tensor_end.end());
    ttsl::Span<const T> step_vec(step.begin(), step.end());
    return slice<T>(
        input_tensor, start, end, step_vec, memory_config_arg, optional_output_tensor, pad_value, sub_core_grids);
}

template <typename T>
ttnn::Tensor slice(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor_start,
    const ttnn::Tensor& output_tensor_end,
    const std::optional<ttsl::SmallVector<T>>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value,
    const std::optional<uint32_t>& slice_dim,
    const std::optional<uint32_t>& num_devices,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    TT_FATAL(
        output_tensor_start.logical_shape().rank() == 1,
        "The start tensor for slicing must be in 1D shape, but got {}D",
        output_tensor_start.logical_shape().rank());
    TT_FATAL(
        output_tensor_end.logical_shape().rank() == 1,
        "The end tensor for slicing must be in 1D shape, but got {}D",
        output_tensor_end.logical_shape().rank());

    // Check if we can use the device-only tensor args path
    bool use_device_only_path = true;

    // Check if layout is supported (only TILE layout for now)
    if (input_tensor.layout() != Layout::TILE) {
        use_device_only_path = false;
    }

    // Check if step > 1 (only step=1 supported for now)
    if (step.has_value()) {
        for (auto s : step.value()) {
            if (s != 1) {
                use_device_only_path = false;
                break;
            }
        }
    }

    // Validate tensors are on device for both paths
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device for tensor args slice");

    auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config()
                                                            : memory_config_arg.value_or(input_tensor.memory_config());

    if (use_device_only_path) {
        // Validate required parameters for device-only path
        TT_FATAL(
            slice_dim.has_value() && num_devices.has_value(),
            "slice_dim and num_devices must be provided for device-only tensor args slice");

        TT_FATAL(
            output_tensor_start.storage_type() == StorageType::DEVICE,
            "Start tensor must be on device for tensor args slice");
        TT_FATAL(
            output_tensor_end.storage_type() == StorageType::DEVICE,
            "End tensor must be on device for tensor args slice");

        // Implicit inheritance from an ND-sharded input: rescale its shard shape to the per-device output,
        // as the list-bounds overload does. Invalid slice_dim / num_devices are left to the device op.
        const auto& input_shape = input_tensor.logical_shape();
        const uint32_t input_rank = input_shape.rank();
        if (!memory_config_arg.has_value() && !optional_output_tensor.has_value() &&
            detail::is_nd_sharded(memory_config) && input_rank >= 2 && slice_dim.value() < input_rank &&
            num_devices.value() > 0) {
            const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
            ttsl::SmallVector<uint32_t> output_dims(input_shape.cbegin(), input_shape.cend());
            output_dims[slice_dim.value()] /= num_devices.value();
            output_dims[input_rank - 2] =
                std::max(tt::round_up(output_dims[input_rank - 2], tile_shape[0]), tile_shape[0]);
            output_dims[input_rank - 1] =
                std::max(tt::round_up(output_dims[input_rank - 1], tile_shape[1]), tile_shape[1]);
            memory_config = detail::rescale_inherited_nd_config(
                memory_config, input_tensor, ttnn::Shape(output_dims), Layout::TILE);
        }

        // Create dummy shapes for SliceDeviceOperation (will be ignored when use_tensor_args=true)
        ttsl::SmallVector<uint32_t> dummy_shape(input_rank, 0);
        ttsl::SmallVector<uint32_t> dummy_step_shape(input_rank, 1);
        ttnn::Shape dummy_start(dummy_shape);
        ttnn::Shape dummy_end(dummy_shape);
        ttnn::Shape dummy_step(dummy_step_shape);

        // Use slice device operation with tensor args flag
        std::optional<Tensor> start_opt = output_tensor_start;
        std::optional<Tensor> end_opt = output_tensor_end;

        auto res = ttnn::prim::qsr::slice(
            input_tensor,
            dummy_start,
            dummy_end,
            dummy_step,
            memory_config,
            /*use_tensor_args*/ true,
            start_opt,
            end_opt,
            slice_dim,
            num_devices,
            sub_core_grids,
            optional_output_tensor);
        return res;
    }  // convert the Tensor to Vector
    std::vector<T> output_tensor_start_vector = output_tensor_start.to_vector<T>();
    std::vector<T> output_tensor_end_vector = output_tensor_end.to_vector<T>();

    // convert the Vector to Span
    ttsl::Span<const T> output_tensor_start_span(output_tensor_start_vector.data(), output_tensor_start_vector.size());
    ttsl::Span<const T> output_tensor_end_span(output_tensor_end_vector.data(), output_tensor_end_vector.size());

    // generate the step value if it is not provided
    ttsl::SmallVector<T> step_value = step.value_or(ttsl::SmallVector<T>(output_tensor_start_span.size(), 1));

    return slice<T>(
        input_tensor,
        output_tensor_start_span,
        output_tensor_end_span,
        ttsl::Span<const T>(step_value),
        memory_config_arg,
        optional_output_tensor,
        pad_value,
        sub_core_grids);
}

// Template instantiations for ttnn::slice
template ttnn::Tensor slice<int32_t>(
    const ttnn::Tensor& input_tensor,
    ttsl::Span<const int32_t> begins,
    ttsl::Span<const int32_t> ends,
    ttsl::Span<const int32_t> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value,
    const std::optional<CoreRangeSet>& sub_core_grids);

template ttnn::Tensor slice<uint32_t>(
    const ttnn::Tensor& input_tensor,
    ttsl::Span<const uint32_t> begins,
    ttsl::Span<const uint32_t> ends,
    ttsl::Span<const uint32_t> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value,
    const std::optional<CoreRangeSet>& sub_core_grids);

// Template instantiations for std::array version
template ttnn::Tensor slice<uint32_t, 4>(
    const ttnn::Tensor& input_tensor,
    const std::array<uint32_t, 4>& output_tensor_start,
    const std::array<uint32_t, 4>& output_tensor_end,
    const std::array<uint32_t, 4>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value,
    const std::optional<CoreRangeSet>& sub_core_grids);

// Template instantiations for Tensor version
template ttnn::Tensor slice<uint32_t>(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor_start,
    const ttnn::Tensor& output_tensor_end,
    const std::optional<ttsl::SmallVector<uint32_t>>& step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value,
    const std::optional<uint32_t>& slice_dim,
    const std::optional<uint32_t>& num_devices,
    const std::optional<CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::experimental::quasar
