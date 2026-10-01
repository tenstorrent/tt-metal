// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_codegen_supported.hpp"

#include <cstdint>

#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>

#include "gather_codegen_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::gather {

namespace {

// The one tile geometry the kernels implement. Every reader walks a fixed 2x2 grid of 16x16 faces
// and make_tile_cb() leaves CBFormatDescriptor::tile unset, so the kernels address a 32x32 tile
// whatever the tensor's spec claims. Tile::operator== compares shapes only, hence the explicit
// transpose flags.
bool has_default_tile(const Tensor& tensor) {
    const auto& tile = tensor.tensor_spec().tile();
    return tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH &&
           !tile.get_transpose_within_face() && !tile.get_transpose_of_faces();
}

}  // namespace

bool supported_execution_controls(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    // Every codegen kernel sizes its output CB and its per-tile transfer from the output tensor's
    // own aligned TILE page, which is only well-defined for a non-sharded interleaved buffer. Both
    // the requested memory config and a caller-supplied destination decide that placement, so both
    // are checked here.
    //
    // sub_core_grids is deliberately absent: all three gather factories thread it into their work
    // split (split_gather_work), so the caller's core reservation is honoured rather than widened.
    const auto& output_mem_config = memory_config.has_value() ? memory_config.value() : input_tensor.memory_config();
    if (output_mem_config.is_sharded()) {
        return false;
    }
    if (!optional_output_tensor.has_value()) {
        return true;
    }
    const auto& out = optional_output_tensor.value();
    if (out.memory_config().is_sharded()) {
        return false;
    }
    // A caller-supplied destination keeps its own spec through compute_output_specs(), so it -- not
    // the input -- is what make_tile_cb() cuts the output CB page from and what the writer transfers
    // per tile. The readers emit a full 32x32 tile of elements at a stride derived from that page,
    // and the L1 feasibility check budgets the output page as equal to the input page, so only the
    // spec the op would have created for itself is in contract.
    return out.layout() == Layout::TILE && out.dtype() == input_tensor.dtype() && has_default_tile(out);
}

bool supported_by_codegen(const Tensor& input_tensor, int8_t dim, const Tensor& input_index_tensor) {
    // The codegen kernels address purely in tile-page units (Ht/Wt_input/Wt_index) and have no
    // stick/row-major mode.
    if (input_tensor.layout() != Layout::TILE || input_index_tensor.layout() != Layout::TILE) {
        return false;
    }
    // Layout::TILE also admits tiny and transposed tiles, which the kernels cannot address. The
    // geometry helper additionally divides BOTH padded widths by the input tile's width, so Wt_index
    // is only right when the two tensors carry the same tile.
    if (!has_default_tile(input_tensor) || !has_default_tile(input_index_tensor)) {
        return false;
    }
    // Only bfloat16 input is in scope. Wider input dtypes reach these kernels, if at all, through a
    // host decode->gather->encode composition rather than directly.
    if (input_tensor.dtype() != DataType::BFLOAT16) {
        return false;
    }
    // These kernels have no sharded path; fall back to native, which handles the full memory-config
    // spectrum through its own TensorAccessor-based kernels.
    if (input_tensor.memory_config().is_sharded() || input_index_tensor.memory_config().is_sharded()) {
        return false;
    }

    // ttnn::gather() runs this gate ahead of its own dim range check, so an out-of-range dim must
    // not be indexed here; let it fall through to native's error.
    const auto rank = static_cast<int32_t>(input_tensor.logical_shape().rank());
    if (dim < -rank || dim >= rank) {
        return false;
    }
    const int32_t gather_axis = dim < 0 ? dim + rank : dim;

    // A zero-extent index has no tile-rows to split, so the work split has nothing to divide, and
    // ttnn::gather()'s own early exit only covers a rank-0 shape.
    if (input_tensor.logical_shape().volume() == 0 || input_index_tensor.logical_shape().volume() == 0) {
        return false;
    }

    // An index element must be able to name any position along the gathered axis. The kernels read
    // the index at its own byte width, so a uint16 index on a longer axis cannot express the upper
    // positions at all. Positions are 0-based, so the limit is the axis LENGTH, not the max value:
    // 65536 addresses 0..65535 and fits.
    constexpr uint32_t kUint16MaxAxisLength = 65536;
    if (input_index_tensor.dtype() == DataType::UINT16 &&
        input_tensor.logical_shape()[gather_axis] > kUint16MaxAxisLength) {
        return false;
    }

    // Device-resource feasibility. Both dimension-scaled CB plans (row-buffered/tiled: Wt_input and
    // gather_output_cb_tiles(Wt_index) tile pages; streaming: chunk_tiles input pages) scale down to the streaming
    // floor, so a call is feasible exactly when that floor fits per-core L1. Only answerable with a
    // real device behind the tensors; the prim's validation step raises native's structural error
    // for anything else. Measured against the device's static L1 window, never its live occupancy,
    // which is what keeps this predicate's answer identical at the router and at validate.
    const bool on_device = input_tensor.storage_type() == StorageType::DEVICE &&
                           input_index_tensor.storage_type() == StorageType::DEVICE &&
                           input_tensor.buffer() != nullptr && input_index_tensor.buffer() != nullptr;
    return !on_device || ttnn::prim::gather_min_plan_fits_l1(input_tensor, input_index_tensor);
}

bool is_demoted(const Tensor& /*input_tensor*/, int8_t /*dim*/, const Tensor& /*input_index_tensor*/) {
    // The perf gate demotes NOTHING, so ttnn::gather() runs codegen wherever supported_by_codegen()
    // admits it -- including one class that is accepted rather than demoted: a gather whose logical
    // height is a single row runs ~2% slower than the native prim. These kernels work in whole
    // tiles while native's scale with real element rows, so at height 1 native touches 1/32 of a
    // tile the readers still walk end to end, and its 32x smaller workload just edges ahead.
    //
    // Accepted because no predicate over the attributes isolates it. [1, 32768] and
    // [1, 1, 32, 32768] agree on dtype, layout, dim, Wt_input and Wt_index, and both are full
    // width, yet they measure 1.03x and 0.21x: a condition on width, or on the index/input tile
    // ratio, would demote a case nearly five times faster than native to spare a 2% one. Only the
    // logical height separates them, and where between height 1 and the heights that win it turns
    // is unmeasured.
    //
    // The signature stays so the routing expression's `supported && !demoted` wiring is identical
    // whether or not a demotion is ever found; a future demotion belongs here as a condition over
    // the tensor attributes, not as a shape-tuple carve-out.
    //
    // One in-scope configuration -- input [1,1,32,64], dim=-2, index [1,1,16,64], bfloat16, TILE --
    // beats the native prim but sits below parity with the same descriptor replayed through
    // generic_op, and is accepted rather than demoted. It post-transforms to Ht=2, Wt_input=1,
    // Wt_index=1, i.e. the smallest work the row-buffered factory can be given (two tile-rows, two
    // cores), where the measured difference is fixed per-launch cost and not anything the descriptor
    // chooses. Demoting a case that beats native would route it to the slower path.
    return false;
}

}  // namespace ttnn::operations::data_movement::gather
