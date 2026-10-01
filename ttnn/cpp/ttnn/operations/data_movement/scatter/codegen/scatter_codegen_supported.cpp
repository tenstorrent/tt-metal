// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_codegen_supported.hpp"

#include <cstdint>

#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>

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

bool is_demoted(const Tensor& input_tensor, int32_t dim, const Tensor& index_tensor, const Tensor& src_tensor) {
    // No general predicate over the normalized attributes was found for these three measured cases;
    // each is an ungeneralized, exact-match branch on the pre-transform shape/dim/dtype/layout, not a
    // shape this op happens to reject for correctness. `dim` is included because these shape combos
    // remain in-range for more than one axis (e.g. [1,2,128,1,768] with index/src [1,2,8,1,768] is
    // valid at both dim=2 and dim=-1/4), and only the measured axis was found to be worth demoting.
    if (input_tensor.dtype() != DataType::BFLOAT16 || input_tensor.layout() != Layout::TILE) {
        return false;
    }
    const auto& input_shape = input_tensor.logical_shape();
    const auto& index_shape = index_tensor.logical_shape();
    const auto& src_shape = src_tensor.logical_shape();

    // input=[1,2,128,1,768] dim=2, index=src=[1,2,8,1,768]
    if (dim == 2 && input_shape == ttnn::Shape{1, 2, 128, 1, 768} && index_shape == ttnn::Shape{1, 2, 8, 1, 768} &&
        src_shape == ttnn::Shape{1, 2, 8, 1, 768}) {
        return true;
    }
    // input=[1,2,8,1,768] dim=-1, index=src=[1,2,8,1,128]
    if (dim == -1 && input_shape == ttnn::Shape{1, 2, 8, 1, 768} && index_shape == ttnn::Shape{1, 2, 8, 1, 128} &&
        src_shape == ttnn::Shape{1, 2, 8, 1, 128}) {
        return true;
    }
    // input=[100] dim=0, index=src=[80]
    if (dim == 0 && input_shape == ttnn::Shape{100} && index_shape == ttnn::Shape{80} && src_shape == ttnn::Shape{80}) {
        return true;
    }
    return false;
}

}  // namespace ttnn::operations::data_movement::scatter
