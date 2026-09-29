// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat/codegen/repeat_codegen_supported.hpp"

#include <algorithm>
#include <iterator>
#include <optional>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/data_movement/repeat/codegen/repeat_codegen_program_factory.hpp"
#include "ttnn/operations/data_movement/repeat/device/repeat_utils.hpp"

namespace ttnn::operations::data_movement::repeat_codegen {

namespace {

using tt::tt_metal::BufferType;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::ShardOrientation;
using tt::tt_metal::TensorMemoryLayout;

bool is_sub_tile(const ttnn::Shape& shape) {
    return shape[-2] % tt::constants::TILE_HEIGHT != 0 || shape[-1] % tt::constants::TILE_WIDTH != 0;
}

// ttnn does not support a bfloat16 ROW_MAJOR tensor whose last dim is a single element, and native
// refuses it, so a row-major leg over one is out of scope.
bool row_major_stick_ok(DataType dtype, const ttnn::Shape& shape) {
    return dtype != DataType::BFLOAT16 || shape[-1] >= 2;
}

MemoryConfig interleaved_in(BufferType buffer_type) {
    return MemoryConfig{TensorMemoryLayout::INTERLEAVED, buffer_type};
}

// The spec of a buffer of `shape`/`layout` at `memory_config`, with `ref`'s dtype and tile.
tt::tt_metal::TensorSpec spec_like(
    const Tensor& ref, const ttnn::Shape& shape, Layout layout, const MemoryConfig& memory_config) {
    const tt::tt_metal::PageConfig page_config = layout == Layout::TILE
                                                     ? tt::tt_metal::PageConfig(layout, ref.tensor_spec().tile())
                                                     : tt::tt_metal::PageConfig(layout);
    return tt::tt_metal::TensorSpec(shape, tt::tt_metal::TensorLayout(ref.dtype(), page_config, memory_config));
}

// Per-bank bytes a buffer of `spec` takes out of L1 on `ref`'s device; zero when it lives in DRAM.
uint64_t l1_bytes_per_bank(const Tensor& ref, const tt::tt_metal::TensorSpec& spec) {
    if (spec.memory_config().buffer_type() != BufferType::L1) {
        return 0;
    }
    const auto& allocator = ref.device()->allocator();
    return spec.compute_consumed_memory_bytes_per_bank(
        allocator->get_alignment(BufferType::L1), allocator->get_num_banks(BufferType::L1));
}

// Both ROW_MAJOR branches page one stick per CB slot, and a stick scales with the tensor's width. The
// factory scales its depth down to the L1 it finds free; reject upfront any leg where even the
// smallest viable CB would not fit, so an oversized repeat cleanly routes to native (which streams the
// same output without a stick-sized CB) instead of failing circular buffer allocation.
//
// The budget is the static L1 window less `committed_l1`, the per-bank bytes of the L1 buffers this
// call keeps alive alongside the leg. Live occupancy is never read here: this gate runs at routing and
// again on a program-cache miss, with the op's own outputs allocated in between, and the two answers
// must agree.
//
// The slot is sized from the leg's own input and output specs, exactly as the factory sizes it from
// their buffers.
bool rm_leg_fits_in_l1(
    const Tensor& input,
    const tt::tt_metal::TensorSpec& leg_input,
    const tt::tt_metal::TensorSpec& leg_output,
    uint64_t committed_l1) {
    const uint32_t slot = ttnn::prim::rm_slot_bytes(
        ttnn::prim::spec_aligned_page_bytes(input, leg_input), ttnn::prim::spec_aligned_page_bytes(input, leg_output));
    const uint64_t window = ttnn::prim::static_l1_window(input);
    return committed_l1 < window && ttnn::prim::plan_rm_cb(slot, window - committed_l1).has_value();
}

// The host-side page map that feeds the codegen prim derives Ht/Wt from the 32x32
// constants, so an off-default tile shape gives the kernels both a page count and a
// page size the buffer does not have. A transposed tile keeps both but swizzles the
// datums within the page, and compute_output_specs() derives the output tile from the
// layout alone, so the copied pages would come back labelled untransposed. A ROW_MAJOR
// page is a whole stick and never consults the tile, so this constrains TILE only.
bool tile_geometry_ok(const Tensor& input) {
    if (input.layout() != ttnn::TILE_LAYOUT) {
        return true;
    }
    const auto& tile = input.tensor_spec().tile();
    return tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH &&
           !tile.get_transpose_within_face() && !tile.get_transpose_of_faces();
}

}  // namespace

bool shard_spec_is_page_identical(const MemoryConfig& memory_config, const ttnn::Shape& logical_shape, Layout layout) {
    if (!memory_config.is_sharded()) {
        return true;
    }
    const auto& shard_spec = memory_config.shard_spec();
    if (!shard_spec.has_value() || logical_shape.rank() < 2) {
        return false;
    }
    const bool is_tile = layout == Layout::TILE;
    uint64_t height = 1;
    for (int32_t i = 0; i < static_cast<int32_t>(logical_shape.rank()) - 2; ++i) {
        height *= logical_shape[i];
    }
    uint64_t rows = logical_shape[-2];
    uint64_t width = logical_shape[-1];
    if (is_tile) {
        rows = tt::round_up(rows, static_cast<uint64_t>(tt::constants::TILE_HEIGHT));
        width = tt::round_up(width, static_cast<uint64_t>(tt::constants::TILE_WIDTH));
    }
    height *= rows;

    const uint64_t shard_h = shard_spec->shape[0];
    const uint64_t shard_w = shard_spec->shape[1];
    if (shard_h == 0 || shard_w == 0) {
        return false;
    }
    if (is_tile && (shard_h % tt::constants::TILE_HEIGHT != 0 || shard_w % tt::constants::TILE_WIDTH != 0)) {
        return false;
    }
    // A partial final shard changes the page-to-core map away from the one the accessor computes.
    if (height % shard_h != 0 || width % shard_w != 0) {
        return false;
    }
    const uint64_t height_shards = height / shard_h;
    const uint64_t width_shards = width / shard_w;
    // A ROW_MAJOR page is the whole stick only when the shard spans the whole row.
    if (!is_tile && width_shards != 1) {
        return false;
    }

    const uint64_t num_cores = shard_spec->grid.num_cores();
    uint64_t active = 0;
    switch (memory_config.memory_layout()) {
        case TensorMemoryLayout::HEIGHT_SHARDED:
            if (width_shards != 1) {
                return false;
            }
            active = height_shards;
            break;
        case TensorMemoryLayout::WIDTH_SHARDED:
            if (height_shards != 1) {
                return false;
            }
            active = width_shards;
            break;
        case TensorMemoryLayout::BLOCK_SHARDED: {
            const CoreRange bbox = shard_spec->grid.bounding_box();
            if (bbox.size() != num_cores) {
                return false;
            }
            const CoreCoord extent = bbox.grid_size();
            const bool row_major = shard_spec->orientation == ShardOrientation::ROW_MAJOR;
            const uint64_t grid_h = row_major ? extent.y : extent.x;
            const uint64_t grid_w = row_major ? extent.x : extent.y;
            if (grid_h != height_shards || grid_w != width_shards) {
                return false;
            }
            active = height_shards * width_shards;
            break;
        }
        default: return false;
    }
    return active <= num_cores;
}

bool output_matches_input_page(const Tensor& input, const Tensor& output) {
    if (output.dtype() != input.dtype() || output.layout() != input.layout()) {
        return false;
    }
    if (input.layout() != Layout::TILE) {
        return true;
    }
    const auto& in_tile = input.tensor_spec().tile();
    const auto& out_tile = output.tensor_spec().tile();
    return out_tile.get_height() == in_tile.get_height() && out_tile.get_width() == in_tile.get_width() &&
           out_tile.get_transpose_within_face() == in_tile.get_transpose_within_face() &&
           out_tile.get_transpose_of_faces() == in_tile.get_transpose_of_faces();
}

bool needs_row_major_round_trip(const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims) {
    const auto& shape = input.logical_shape();
    const uint32_t ndim = shape.rank();
    if (input.layout() != ttnn::TILE_LAYOUT || ndim < 2 || repeat_dims.size() != ndim || !is_sub_tile(shape)) {
        return false;
    }
    return repeat_dims[ndim - 2] > 1 || repeat_dims[ndim - 1] > 1;
}

CodegenLegPlan plan_codegen_legs(
    const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims, const MemoryConfig& output_mem_config) {
    const auto& shape = input.logical_shape();
    const uint32_t ndim = std::min<uint32_t>(shape.rank(), repeat_dims.size());
    const MemoryConfig& input_mc = input.memory_config();

    CodegenLegPlan plan;
    plan.round_trip = needs_row_major_round_trip(input, repeat_dims);
    // A page-identical sharded input is read where it lies. Anything else unshards once, so a single
    // hop feeds every leg; the untilize of the round trip is interleaved-only.
    plan.unshard_input =
        input_mc.is_sharded() && (plan.round_trip || !shard_spec_is_page_identical(input_mc, shape, input.layout()));
    // A repeated shape does not tile the input's shard spec, so intermediates are interleaved.
    plan.intermediate_mc = interleaved_in(plan.unshard_input ? BufferType::DRAM : input_mc.buffer_type());

    auto out_shape = shape;
    for (uint32_t d = 0; d < ndim; ++d) {
        out_shape[d] *= repeat_dims[d];
    }
    plan.final_in_place =
        !plan.round_trip && shard_spec_is_page_identical(output_mem_config, out_shape, input.layout());
    plan.final_mc = plan.final_in_place ? output_mem_config : interleaved_in(output_mem_config.buffer_type());

    for (uint32_t d = 0; d < ndim; ++d) {
        if (repeat_dims[d] > 1) {
            plan.rep_dims.push_back(d);
        }
    }
    // Row-major legs repeat the stick first, so tiny multi-dim repeats do not carry a wider
    // intermediate through the per-stick work.
    if (plan.round_trip || input.layout() == ttnn::ROW_MAJOR_LAYOUT) {
        std::reverse(plan.rep_dims.begin(), plan.rep_dims.end());
    }
    return plan;
}

bool supported_by_codegen(
    const Tensor& input, uint32_t rep_dim, uint32_t num_repeats, const MemoryConfig& output_mem_config) {
    if (!shard_spec_is_page_identical(input.memory_config(), input.logical_shape(), input.layout())) {
        return false;
    }
    if (!tile_geometry_ok(input)) {
        return false;
    }
    const auto& shape = input.logical_shape();
    if (shape.rank() != 4 || rep_dim >= 4 || num_repeats == 0) {
        return false;
    }
    if (num_repeats == 1) {
        return true;
    }
    if (input.layout() == ttnn::ROW_MAJOR_LAYOUT) {
        if (input.dtype() == DataType::BFLOAT8_B || !row_major_stick_ok(input.dtype(), shape)) {
            return false;
        }
        if (input.storage_type() != ttnn::StorageType::DEVICE) {
            // Not yet on device (e.g. host-side probing); nothing to bound against.
            return true;
        }
        // Only this step's own input and output are known here; the whole-call gate, which also
        // counts the call's other live intermediates, subtracts at least as much.
        auto out_shape = shape;
        out_shape[rep_dim] *= num_repeats;
        const auto& in_spec = input.tensor_spec();
        const auto out_spec = spec_like(input, out_shape, Layout::ROW_MAJOR, output_mem_config);
        const uint64_t committed = l1_bytes_per_bank(input, in_spec) + l1_bytes_per_bank(input, out_spec);
        return rm_leg_fits_in_l1(input, in_spec, out_spec, committed);
    }
    if (input.layout() != ttnn::TILE_LAYOUT) {
        return false;
    }
    // The prim copies whole tile pages, which is exact along H or W only when that axis is tile-aligned.
    if (rep_dim == 3) {
        return shape[-1] % tt::constants::TILE_WIDTH == 0;
    }
    if (rep_dim == 2) {
        return shape[-2] % tt::constants::TILE_HEIGHT == 0;
    }
    return true;
}

bool supported_by_codegen(
    const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims, const MemoryConfig& output_mem_config) {
    if (!tile_geometry_ok(input)) {
        return false;
    }
    const auto& shape = input.logical_shape();
    const uint32_t ndim = shape.rank();
    // The codegen kernels index pages through a fixed 4D page map, so a rank > 4
    // input has no path here.
    if (repeat_dims.size() != ndim || ndim < 2 || ndim > 4) {
        return false;
    }
    if (std::none_of(repeat_dims.cbegin(), repeat_dims.cend(), [](uint32_t r) { return r > 1; })) {
        return false;
    }
    // The legs repeat only the dims above 1, so a zero elsewhere would be dropped instead of emptying
    // the output.
    if (std::any_of(repeat_dims.cbegin(), repeat_dims.cend(), [](uint32_t r) { return r == 0; })) {
        return false;
    }
    // The router resolves a sharded output's spec before asking; one still missing here has no page
    // grid for the final leg to write.
    if (output_mem_config.is_sharded() && !output_mem_config.shard_spec().has_value()) {
        return false;
    }
    // A sharded input that is not page-identical is unsharded once up front, so any shard spec is
    // served; only the tile/stick rules below constrain the call.
    const bool round_trip = needs_row_major_round_trip(input, repeat_dims);
    if (input.layout() == ttnn::TILE_LAYOUT && !round_trip) {
        // Only outer axes, or tile-aligned H/W, are repeated: plain tile-page copies.
        return true;
    }
    if (input.layout() != ttnn::TILE_LAYOUT && input.layout() != ttnn::ROW_MAJOR_LAYOUT) {
        return false;
    }
    // Every leg below reads the input's stick width first, so the one-stick rule applies to the round
    // trip's row-major legs as well.
    if (input.dtype() == DataType::BFLOAT8_B || !row_major_stick_ok(input.dtype(), shape)) {
        return false;
    }
    if (input.storage_type() != ttnn::StorageType::DEVICE) {
        // Not yet on device (e.g. host-side probing); nothing to bound against.
        return true;
    }
    // Every leg runs row-major, in the order the router executes them. Each leg's CB shares L1 with
    // the input, the round trip's untilized copy and every leg output so far; all of them are counted
    // as live for the whole call, which never undercounts whatever the allocator frees in between.
    const CodegenLegPlan plan = plan_codegen_legs(input, repeat_dims, output_mem_config);
    uint64_t committed = l1_bytes_per_bank(input, input.tensor_spec());
    // What the first leg reads: the input where it lies, its interleaved DRAM copy, or the round trip's
    // untilized copy.
    tt::tt_metal::TensorSpec leg_in = input.tensor_spec();
    if (plan.unshard_input) {
        leg_in = spec_like(input, shape, input.layout(), interleaved_in(BufferType::DRAM));
    }
    if (plan.round_trip) {
        leg_in = spec_like(input, shape, Layout::ROW_MAJOR, plan.intermediate_mc);
        committed += l1_bytes_per_bank(input, leg_in);
    }
    for (size_t i = 0; i < plan.rep_dims.size(); ++i) {
        const uint32_t d = plan.rep_dims[i];
        auto out_shape = leg_in.logical_shape();
        out_shape[d] *= repeat_dims[d];
        const MemoryConfig& leg_mc = i + 1 == plan.rep_dims.size() ? plan.final_mc : plan.intermediate_mc;
        const auto leg_out = spec_like(input, out_shape, Layout::ROW_MAJOR, leg_mc);
        committed += l1_bytes_per_bank(input, leg_out);
        if (!rm_leg_fits_in_l1(input, leg_in, leg_out, committed)) {
            return false;
        }
        leg_in = leg_out;
    }
    return true;
}

// A ROW_MAJOR shard narrower than the row makes each page a partial stick, which the codegen page map
// cannot address, so the codegen route unshards the whole input to DRAM before its legs and, for a
// sharded output, reshards after them. When exactly one axis is repeated and native's sharded
// predicate accepts the call, native instead repeats each shard where it lies in one program, so the
// codegen route pays two or three extra full-tensor moves for the same work. With two or more
// repeated axes native unshards up front too, and the two routes compete on equal terms.
bool is_demoted(
    const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims, const MemoryConfig& output_mem_config) {
    const auto& input_mc = input.memory_config();
    if (input.layout() != Layout::ROW_MAJOR || !input_mc.is_sharded() ||
        shard_spec_is_page_identical(input_mc, input.logical_shape(), Layout::ROW_MAJOR)) {
        return false;
    }
    const auto repeated = std::count_if(repeat_dims.cbegin(), repeat_dims.cend(), [](uint32_t r) { return r != 1; });
    if (repeated != 1) {
        return false;
    }
    const auto* const it = std::find_if(repeat_dims.cbegin(), repeat_dims.cend(), [](uint32_t r) { return r != 1; });
    const auto dim = static_cast<int32_t>(std::distance(repeat_dims.cbegin(), it));
    return repeat::is_native_repeat_sharding(
        input.tensor_spec(), std::optional<MemoryConfig>{output_mem_config}, dim, *it);
}

}  // namespace ttnn::operations::data_movement::repeat_codegen
