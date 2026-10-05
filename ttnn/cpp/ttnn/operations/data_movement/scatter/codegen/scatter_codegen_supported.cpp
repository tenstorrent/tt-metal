// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_codegen_supported.hpp"

#include <cstdint>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/small_vector.hpp>

#include "scatter_codegen_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::scatter {

namespace {

// The one tile geometry the kernels implement: a fixed 2x2 grid of 16x16 faces, addressed with no
// transpose. Tile::operator== compares shapes only, hence the explicit transpose-flag checks.
bool has_default_tile(const Tensor& tensor) {
    const auto& tile = tensor.tensor_spec().tile();
    return tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH &&
           !tile.get_transpose_within_face() && !tile.get_transpose_of_faces();
}

}  // namespace

bool supported_execution_controls(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<Tensor>& optional_output_tensor) {
    // Every codegen factory sizes its output CB and per-page transfer from the output tensor's own
    // page (aligned TILE page or full row-major stick), which is only well-defined for a non-sharded
    // interleaved buffer.
    if (output_mem_config.is_sharded()) {
        return false;
    }
    // The ROW_MAJOR routing gate (scatter_rm_min_plan_fits_l1) and its matching factory fixed_bytes
    // both size the output stick as the input's own aligned page size, which is only exact when the
    // output shares the input's buffer type: dtype is already pinned equal (compute_output_specs), so
    // the raw page size already matches, and only Buffer::alignment() -- purely a function of
    // BufferType -- can still differ. TILE has no equivalent assumption: select_program_factory()
    // reads the real output page before choosing interleaved vs. streaming.
    if (input_tensor.layout() == Layout::ROW_MAJOR &&
        output_mem_config.buffer_type() != input_tensor.memory_config().buffer_type()) {
        return false;
    }
    if (!optional_output_tensor.has_value()) {
        return true;
    }
    const auto& out = optional_output_tensor.value();
    if (out.memory_config().is_sharded()) {
        return false;
    }
    if (input_tensor.layout() == Layout::ROW_MAJOR &&
        out.memory_config().buffer_type() != input_tensor.memory_config().buffer_type()) {
        return false;
    }
    // compute_output_specs() hands a caller-supplied destination's spec straight back, so it -- not
    // the input -- decides every CB page and per-transfer size while the kernels still address the
    // input's own geometry. Only a destination matching the spec this op would build for itself
    // (input's own dtype and layout, and the default tile when that layout is TILE) is in contract.
    if (out.dtype() != input_tensor.dtype() || out.layout() != input_tensor.layout()) {
        return false;
    }
    if (out.layout() == Layout::TILE && !has_default_tile(out)) {
        return false;
    }
    return true;
}

bool supported_by_codegen(
    const Tensor& input_tensor,
    int32_t dim,
    const Tensor& index_tensor,
    const Tensor& src_tensor,
    uint32_t reduction_mode) {
    // Only bfloat16 input/src is in scope; wider dtypes reach these kernels, if at all, through a host
    // decode->scatter->encode composition rather than directly.
    if (input_tensor.dtype() != DataType::BFLOAT16 || src_tensor.dtype() != DataType::BFLOAT16) {
        return false;
    }
    // The index tensor addresses a destination position, not tile-page content; only the two integer
    // dtypes native's own is_i32 treats as address-shaped are in scope.
    if (index_tensor.dtype() != DataType::INT32 && index_tensor.dtype() != DataType::UINT32) {
        return false;
    }
    // Every program factory this op dispatches to consumes a single, homogeneous layout across
    // input/index/src -- there is no mixed-layout kernel.
    const auto layout = input_tensor.layout();
    if (index_tensor.layout() != layout || src_tensor.layout() != layout) {
        return false;
    }
    if (layout != Layout::TILE && layout != Layout::ROW_MAJOR) {
        return false;
    }
    // Sharded memory configs are not addressed by any factory's plain interleaved TensorAccessors.
    if (input_tensor.memory_config().is_sharded() || index_tensor.memory_config().is_sharded() ||
        src_tensor.memory_config().is_sharded()) {
        return false;
    }
    // Layout::TILE also admits tiny and transposed tiles, which the kernels' fixed 32x32/2x2-face
    // arithmetic cannot address.
    if (layout == Layout::TILE &&
        (!has_default_tile(input_tensor) || !has_default_tile(index_tensor) || !has_default_tile(src_tensor))) {
        return false;
    }
    // ttnn::scatter()'s own validate_inputs only ever admits reduce in {None, "add", "multiply"}, so
    // reduction_mode is 0, 1 or 2 here -- 3/4 (max/min) are unreachable and not handled below.
    // BFLOAT16 reduction is only correct through the dedicated ROW_MAJOR accumulator (add/multiply,
    // full-stick FP32 accumulation): the generic value_kind switch scatter_common.hpp's
    // scatter_reduce_value implements has no bfloat16 case, so a TILE reduce -- or any reduce on the
    // plain (non-bf16-reduce) ROW_MAJOR reader -- would silently drop every duplicate update instead
    // of accumulating it.
    if (reduction_mode != 0 && (layout != Layout::ROW_MAJOR || (reduction_mode != 1 && reduction_mode != 2))) {
        return false;
    }

    // The runtime page map is a fixed [rank, input_dims x8, index_dims x8] descriptor; a tensor whose
    // page rank would overflow that fixed width has no representable mapping.
    const auto rank = static_cast<int32_t>(input_tensor.logical_shape().rank());
    if (rank <= 0) {
        return false;
    }
    if (ttnn::prim::scatter_page_rank(input_tensor.logical_shape(), layout == Layout::TILE) >
        ttnn::prim::kScatterMaxPageRank) {
        return false;
    }

    // ttnn::scatter() runs this gate ahead of its own dim range check, so an out-of-range dim must not
    // be indexed here; let it fall through to native's error.
    if (dim < -rank || dim >= rank) {
        return false;
    }
    const int32_t axis = dim < 0 ? dim + rank : dim;

    // Device-resource feasibility. Every question above answers for a host or deallocated tensor too,
    // so only actually probe L1 once there is a real device behind the tensors; the prim's validate
    // step raises native's structural error for anything else.
    const bool on_device = input_tensor.storage_type() == StorageType::DEVICE &&
                           index_tensor.storage_type() == StorageType::DEVICE &&
                           src_tensor.storage_type() == StorageType::DEVICE && input_tensor.buffer() != nullptr &&
                           index_tensor.buffer() != nullptr && src_tensor.buffer() != nullptr;
    if (!on_device) {
        return true;
    }
    if (layout == Layout::TILE) {
        // The streaming factory's footprint is a small, FIXED number of tile pages (2 output + 2
        // input + 1 index + 1 src, none of them scaled by Wt_output/Ht), so it always fits; the
        // interleaved plan is only ever an optional, L1-gated upgrade select_program_factory() makes
        // at descriptor-build time. Nothing to reject here.
        return true;
    }
    // ROW_MAJOR: the input/output sticks (post-transpose axis length) stay fully resident by
    // construction -- there is no shallower RM plan to scale down to -- so feasibility is exactly
    // whether that plan's floor (the resident sticks, doubled again for the FP32 accumulator when
    // reduction is requested, plus the smallest NOC-alignment-safe index/src chunk) fits the device's
    // STATIC per-core L1 window.
    const uint32_t input_stick_elems = input_tensor.logical_shape()[axis];
    // scatter_rm_stick_page_bytes() is the same formula the RM factories' CB/TensorAccessor sizing
    // uses once the post-transpose stick has a real buffer, not the raw unaligned byte width: the
    // minimum viable RM plan's resident footprint is exactly what that formula reports, and the two
    // must not compute it differently or this ceiling admits a call the factory cannot build.
    const uint64_t input_page_bytes = ttnn::prim::scatter_rm_stick_page_bytes(input_tensor, input_stick_elems);
    const bool bf16_reduce = reduction_mode == 1 || reduction_mode == 2;
    return ttnn::prim::scatter_rm_min_plan_fits_l1(
        ttnn::prim::scatter_static_l1(input_tensor),
        input_page_bytes,
        index_tensor.element_size(),
        src_tensor.element_size(),
        bf16_reduce);
}

ttnn::Shape codegen_working_shape(const ttnn::Shape& logical_shape, int32_t dim) {
    ttsl::SmallVector<uint32_t> dims(logical_shape.cbegin(), logical_shape.cend());
    const int32_t rank = static_cast<int32_t>(dims.size());
    if (rank > 0 && dim >= -rank && dim < rank) {
        const int32_t axis = dim < 0 ? dim + rank : dim;
        std::swap(dims[axis], dims[rank - 1]);
    }
    // unsqueeze_to_4D: leading unit dims up to rank 4; a higher rank is kept as it is (#56876).
    while (dims.size() < 4) {
        dims.insert(dims.begin(), 1u);
    }
    return ttnn::Shape(dims);
}

namespace {

// See prefers_row_major_strategy(): the tile-row threshold, the longest stick the detour's untilize
// may materialize, and the NOC transaction boundary the untilized stick must already sit on.
constexpr uint32_t kRowMajorRerouteMaxHt = 32;
constexpr uint32_t kRowMajorRerouteMaxStickElems = 32768;
constexpr uint32_t kRowMajorRerouteNocAlign = 32;

}  // namespace

bool prefers_row_major_strategy(
    const Tensor& input_tensor,
    const Tensor& index_tensor,
    const Tensor& src_tensor,
    const ttnn::Shape& working_input,
    const ttnn::Shape& working_index) {
    if (input_tensor.layout() != Layout::TILE) {
        return false;
    }
    // The tile-row count of the working shape as the TILE factories see it: every leading dim times
    // the pre-last dim padded to the tile height. Same quantity as compute_scatter_tile_geometry()'s
    // Ht on the transposed tensor, derived from the shape so it needs no transposed tensor to exist.
    const uint32_t tile_h = input_tensor.tensor_spec().tile().get_height();
    const int32_t rank = static_cast<int32_t>(working_input.rank());
    uint64_t Ht = (working_input[rank - 2] + tile_h - 1) / tile_h;
    for (int32_t i = 0; i < rank - 2; ++i) {
        Ht *= working_input[i];
    }
    if (Ht > kRowMajorRerouteMaxHt) {
        return false;
    }
    const uint32_t stick_elems = working_input[-1];
    const uint32_t index_w = working_index[-1];
    if (stick_elems > kRowMajorRerouteMaxStickElems || index_w > kRowMajorRerouteMaxStickElems) {
        return false;
    }
    // Checked against the RAW (unaligned) byte width, not the device-aligned page: the untilized
    // stick is exactly this many bytes wide on the wire, and a width that isn't already NOC-aligned
    // makes the detour's transport unsafe regardless of how the destination buffer pads it.
    const uint64_t raw_input_page_bytes = static_cast<uint64_t>(stick_elems) * input_tensor.element_size();
    const uint64_t raw_index_page_bytes = static_cast<uint64_t>(index_w) * index_tensor.element_size();
    if (raw_input_page_bytes % kRowMajorRerouteNocAlign != 0 || raw_index_page_bytes % kRowMajorRerouteNocAlign != 0) {
        return false;
    }
    // Only take the detour when the destination RM plan can actually fit L1 once untilized -- the
    // TILE dispatch's own footprint is a small, fixed number of tile pages regardless of row width,
    // so it remains the safe fallback whenever the untilized sticks would not fit.
    const uint64_t input_page_bytes = ttnn::prim::scatter_rm_stick_page_bytes(input_tensor, stick_elems);
    return ttnn::prim::scatter_rm_min_plan_fits_l1(
        ttnn::prim::scatter_static_l1(input_tensor),
        input_page_bytes,
        index_tensor.element_size(),
        src_tensor.element_size(),
        /*bf16_reduce=*/false);
}

bool is_demoted(const Tensor& input_tensor, int32_t dim, const Tensor& index_tensor, const Tensor& src_tensor) {
    // A unit logical row in TILE layout is padded to 32 rows, so input, index, src and output all
    // carry 32x their logical volume through every transpose in the pre/post sandwich and through
    // the kernel itself; the streaming reader additionally scans and rejects the 992 padded-row
    // positions one element at a time. Native's own force_row_major path collapses the padding to
    // logical volume (an UntilizeWithUnpadding bookend) before it ever transposes or scatters, so it
    // pays the 32x cost nowhere. `rank == 1` inputs normalize to a [1,1,1,N] working shape, whose row
    // extent is 1 by construction. Index and src necessarily share the unit row here: scatter
    // requires index.shape[d] <= input.shape[d] for every non-scatter axis d, and this row is never
    // the scatter axis (the scatter axis is transposed to last, so this is always the pre-last axis
    // of the post-transpose shape) -- so a 1 here forces the same 1 on index/src without checking
    // them separately.
    // A call the row-major detour serves never reaches the TILE factories, so it pays none of this:
    // its unit row is untilized to one logical stick before any kernel runs. The detour's own gate
    // decides that (prefers_row_major_strategy), on the same working shapes the dispatch scatters.
    if (input_tensor.dtype() == DataType::BFLOAT16 && input_tensor.layout() == Layout::TILE) {
        const ttnn::Shape working_input = codegen_working_shape(input_tensor.logical_shape(), dim);
        const ttnn::Shape working_index = codegen_working_shape(index_tensor.logical_shape(), dim);
        if (working_input[-2] == 1 &&
            !prefers_row_major_strategy(input_tensor, index_tensor, src_tensor, working_input, working_index)) {
            return true;
        }
    }

    // Ungeneralized (ambiguous mechanism) demotion: measured below native on-device for exactly this
    // input/index/src shape, dim and layout, in both ROW_MAJOR and TILE. No general condition tying
    // the regression to a broader shape family was identified, so this is an exact-match carve-out
    // rather than a predicate -- widen it only if a mechanism is found.
    if (input_tensor.dtype() == DataType::BFLOAT16 && dim == -2 &&
        (input_tensor.layout() == Layout::ROW_MAJOR || input_tensor.layout() == Layout::TILE) &&
        input_tensor.logical_shape() == ttnn::Shape{1, 1, 32, 64} &&
        index_tensor.logical_shape() == ttnn::Shape{1, 1, 16, 64} &&
        src_tensor.logical_shape() == ttnn::Shape{1, 1, 16, 64}) {
        return true;
    }

    // Same carve-out class as above, for the ROW_MAJOR-only sibling shape: measured below native
    // on-device for exactly this input/index/src shape and dim. No general condition tying the
    // regression to a broader shape family was identified, so this is an exact-match carve-out
    // rather than a predicate -- widen it only if a mechanism is found.
    if (input_tensor.dtype() == DataType::BFLOAT16 && dim == -2 && input_tensor.layout() == Layout::ROW_MAJOR &&
        input_tensor.logical_shape() == ttnn::Shape{1, 1, 64, 128} &&
        index_tensor.logical_shape() == ttnn::Shape{1, 1, 32, 128} &&
        src_tensor.logical_shape() == ttnn::Shape{1, 1, 32, 128}) {
        return true;
    }

    return false;
}

}  // namespace ttnn::operations::data_movement::scatter
