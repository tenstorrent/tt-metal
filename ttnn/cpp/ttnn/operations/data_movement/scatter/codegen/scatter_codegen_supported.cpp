// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_codegen_supported.hpp"

#include <cstdint>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt_stl/small_vector.hpp>

#include "scatter_codegen_program_factory.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

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

// Whether the tensor is padded exactly as its own dtype, layout, tile and placement pad its logical
// shape -- tile rounding and nothing beyond it. A tensor can carry more padding than that
// (tilize_with_val_padding builds one), and only its padded_shape tells.
bool has_standard_padding(const Tensor& tensor) {
    const TensorSpec standard(
        tensor.logical_shape(),
        TensorLayout(tensor.dtype(), PageConfig(tensor.layout(), tensor.tensor_spec().tile()), tensor.memory_config()));
    return standard.padded_shape() == tensor.padded_shape();
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
    // (input's own logical shape, dtype and layout, and the default tile when that layout is TILE)
    // is in contract: every factory sizes its page count and per-core split from the input's shape,
    // so a destination of another shape would be written past its end or left partly unwritten.
    if (out.logical_shape() != input_tensor.logical_shape()) {
        return false;
    }
    if (out.dtype() != input_tensor.dtype() || out.layout() != input_tensor.layout()) {
        return false;
    }
    if (out.layout() == Layout::TILE && !has_default_tile(out)) {
        return false;
    }
    // The TILE writers compute each destination page from the input's tile-rounded geometry, so a
    // destination padded past that would take its tiles in the padding, not in its logical rows.
    if (!has_standard_padding(out)) {
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
    // The factories count pages from padded_shape, while the page map folds the logical shape and
    // compute_output_specs() allocates tile-rounded padding of it: an operand padded further than
    // that has pages no other part of the plan accounts for, and its updates land on the wrong rows.
    if (!has_standard_padding(input_tensor) || !has_standard_padding(index_tensor) ||
        !has_standard_padding(src_tensor)) {
        return false;
    }
    // scatter.cpp's scatter_reduction_mode() throws on any reduce string other than add/multiply
    // before routing reaches this gate, so reduction_mode is one of the three named modes here.
    // BFLOAT16 reduction is only correct through the dedicated ROW_MAJOR accumulator (add/multiply,
    // full-stick FP32 accumulation): the generic readers overwrite and static_assert
    // reduction_mode == 0, so a TILE reduce -- or any reduce on the plain (non-bf16-reduce)
    // ROW_MAJOR reader -- past this gate would fail at JIT compile rather than run.
    if (reduction_mode != kReductionNone &&
        (layout != Layout::ROW_MAJOR || (reduction_mode != kReductionAdd && reduction_mode != kReductionMultiply))) {
        return false;
    }

    // The runtime page map is a fixed [rank, input_dims x8, index_dims x8] descriptor; a tensor whose
    // page rank would overflow that fixed width has no representable mapping.
    const auto rank = static_cast<int32_t>(input_tensor.logical_shape().rank());
    if (rank <= 0) {
        return false;
    }
    // A zero extent anywhere is a divisor somewhere downstream: a zero scatter axis makes a zero-byte
    // stick page the CB sizing divides by (SIGFPE on the host, not an exception), a zero leading extent
    // is baked into the page map the readers fold ordinals through, and a zero index/src stick makes a
    // zero index/src chunk that never advances the RM readers' chunk loop. Native already owns the
    // empty-tensor contract, so every zero-volume operand stays there.
    if (input_tensor.logical_shape().volume() == 0 || index_tensor.logical_shape().volume() == 0 ||
        src_tensor.logical_shape().volume() == 0) {
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

    // Device-resource feasibility. This and every routing helper consulted after this gate
    // (prefers_row_major_strategy, codegen_plan_fits_live_l1) read L1 geometry off the operands'
    // device, so a host-resident or deallocated operand is out of scope outright; native raises its
    // own structural error for it.
    const bool on_device = input_tensor.storage_type() == StorageType::DEVICE &&
                           index_tensor.storage_type() == StorageType::DEVICE &&
                           src_tensor.storage_type() == StorageType::DEVICE && input_tensor.buffer() != nullptr &&
                           index_tensor.buffer() != nullptr && src_tensor.buffer() != nullptr;
    if (!on_device) {
        return false;
    }
    if (layout == Layout::TILE) {
        // The streaming factory's footprint is a small, FIXED number of tile pages (2 output + 2
        // input + 1 index + 1 src, none of them scaled by Wt_output/Ht), so it always fits; the
        // interleaved plan is only ever an optional upgrade select_program_factory() makes at
        // descriptor-build time, when its row-resident footprint fits L1 AND the tile-row count
        // does not underfill the grid. Nothing to reject here.
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
    const bool bf16_reduce = reduction_mode == kReductionAdd || reduction_mode == kReductionMultiply;
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

// See prefers_row_major_strategy(). Both bounds are measured tunings, not derived limits: on
// Blackhole the per-stick split out-ran the TILE factories' per-tile-row split up to 32 tile rows,
// with Ht=64 kept on the TILE path by the same measurement, and sticks past 32768 elements were
// never measured to win untilized -- the chunked RM plan itself would admit far wider ones.
constexpr uint32_t kRowMajorRerouteMaxHt = 32;
constexpr uint32_t kRowMajorRerouteMaxStickElems = 32768;
// The untilized stick's raw byte width must sit on a 32 B boundary: the generator's own routing
// term for the detour, kept so the detour serves exactly the cases it was measured on. Not a
// transport requirement -- the ROW_MAJOR factory serves unaligned sticks for direct ROW_MAJOR
// callers through the same page-aligned accessors.
constexpr uint32_t kRowMajorRerouteNocAlign = 32;

// L1 bytes on the fullest bank that a ROW_MAJOR tensor of `shape` and `dtype` placed per
// `mem_config` takes once allocated; 0 unless it lands in L1. `device_tensor` only names the device.
uint64_t pending_l1_bytes(
    const Tensor& device_tensor, const ttnn::Shape& shape, DataType dtype, const MemoryConfig& mem_config) {
    return get_pending_l1_output_reservation(
        device_tensor, shape, mem_config, dtype, Layout::ROW_MAJOR, /*require_constructible=*/true);
}

// The L1 a ROW_MAJOR factory will find: what is free now, less the buffers this call still
// allocates before the factory runs. Routing and the dispatch both run ahead of
// create_output_tensors(), which is what moves the frontier the factory and the program hash read.
uint64_t rm_live_l1_budget(const Tensor& input_tensor, uint64_t pending_bytes) {
    const uint64_t usable = ttnn::prim::scatter_usable_l1(input_tensor);
    return usable > pending_bytes ? usable - pending_bytes : 0;
}

}  // namespace

bool prefers_row_major_strategy(
    const Tensor& input_tensor,
    const Tensor& index_tensor,
    const Tensor& src_tensor,
    const ttnn::Shape& working_input,
    const ttnn::Shape& working_index,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool transposed_copies_pending) {
    if (input_tensor.layout() != Layout::TILE) {
        return false;
    }
    // The detour hands the untilized operands to the ROW_MAJOR factories, whose output stick is sized
    // from the input's own aligned page; supported_execution_controls() rejects that pairing when the
    // output's buffer type differs, so a detour taken here would only fail in the prim's validate.
    if (output_mem_config.buffer_type() != input_tensor.memory_config().buffer_type()) {
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
    // Checked against the RAW (unaligned) byte width, not the device-aligned page, as the generator's
    // routing term is defined (see kRowMajorRerouteNocAlign).
    const uint64_t raw_input_page_bytes = static_cast<uint64_t>(stick_elems) * input_tensor.element_size();
    const uint64_t raw_index_page_bytes = static_cast<uint64_t>(index_w) * index_tensor.element_size();
    if (raw_input_page_bytes % kRowMajorRerouteNocAlign != 0 || raw_index_page_bytes % kRowMajorRerouteNocAlign != 0) {
        return false;
    }
    // Only take the detour when the untilized RM plan fits the L1 the factory will find: what is
    // free now, less the three untilized operands and the output this call still allocates ahead
    // of it (each counted only if placed in L1). The TILE dispatch's own footprint is a small,
    // fixed number of tile pages regardless of row width, so it remains the safe fallback whenever
    // the sticks would not fit. The untilized operands and the transposed copies they are cut from
    // have the same working shape, so one more charge per operand covers the copies when they are
    // still pending.
    const uint64_t input_page_bytes = ttnn::prim::scatter_rm_stick_page_bytes(input_tensor, stick_elems);
    const uint64_t operand_bytes =
        pending_l1_bytes(input_tensor, working_input, input_tensor.dtype(), input_tensor.memory_config()) +
        pending_l1_bytes(input_tensor, working_index, index_tensor.dtype(), index_tensor.memory_config()) +
        pending_l1_bytes(input_tensor, working_index, src_tensor.dtype(), src_tensor.memory_config());
    const uint64_t pending_bytes =
        (transposed_copies_pending ? 2 * operand_bytes : operand_bytes) +
        pending_l1_bytes(input_tensor, working_input, input_tensor.dtype(), output_mem_config);
    return ttnn::prim::scatter_rm_min_plan_fits_l1(
        rm_live_l1_budget(input_tensor, pending_bytes),
        input_page_bytes,
        index_tensor.element_size(),
        src_tensor.element_size(),
        /*bf16_reduce=*/false);
}

bool codegen_plan_fits_live_l1(
    const Tensor& input_tensor,
    int32_t dim,
    const Tensor& index_tensor,
    const Tensor& src_tensor,
    uint32_t reduction_mode,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    if (input_tensor.layout() != Layout::ROW_MAJOR) {
        return true;
    }
    const int32_t rank = static_cast<int32_t>(input_tensor.logical_shape().rank());
    const int32_t axis = dim < 0 ? dim + rank : dim;
    const ttnn::Shape working_input = codegen_working_shape(input_tensor.logical_shape(), axis);
    const ttnn::Shape working_index = codegen_working_shape(index_tensor.logical_shape(), axis);
    const uint64_t input_page_bytes = ttnn::prim::scatter_rm_stick_page_bytes(input_tensor, working_input[-1]);
    const bool bf16_reduce = reduction_mode == kReductionAdd || reduction_mode == kReductionMultiply;
    // Still to be allocated before the factory runs: the output, and when the scatter axis is not
    // already last, the transposed copy of each operand (in that operand's own placement).
    uint64_t pending_bytes = pending_l1_bytes(input_tensor, working_input, input_tensor.dtype(), output_mem_config);
    if (axis != rank - 1) {
        pending_bytes +=
            pending_l1_bytes(input_tensor, working_input, input_tensor.dtype(), input_tensor.memory_config()) +
            pending_l1_bytes(input_tensor, working_index, index_tensor.dtype(), index_tensor.memory_config()) +
            pending_l1_bytes(input_tensor, working_index, src_tensor.dtype(), src_tensor.memory_config());
    } else if (src_tensor.logical_shape() != index_tensor.logical_shape()) {
        // Nothing is transposed on the last axis, but a source wider than the index is sliced to the
        // index's shape first, and that slice, in the source's own placement, is what the factory sees.
        pending_bytes += pending_l1_bytes(input_tensor, working_index, src_tensor.dtype(), src_tensor.memory_config());
    }
    return ttnn::prim::scatter_rm_min_plan_fits_l1(
        rm_live_l1_budget(input_tensor, pending_bytes),
        input_page_bytes,
        index_tensor.element_size(),
        src_tensor.element_size(),
        bf16_reduce);
}

bool is_demoted(
    const Tensor& input_tensor,
    int32_t dim,
    const Tensor& index_tensor,
    const Tensor& src_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    // supported_by_codegen() has already rejected an out-of-range dim, so the axis is in range here.
    // The check below works on the normalized axis, so `dim=3` and `dim=-1` on a rank-4 tensor are
    // the same case.
    const int32_t rank = static_cast<int32_t>(input_tensor.logical_shape().rank());
    const int32_t axis = dim < 0 ? dim + rank : dim;

    // A unit logical row in TILE layout is padded to 32 rows, so input, index, src and output all
    // carry 32x their logical volume through every transpose in the pre/post sandwich and through
    // the kernel itself; the streaming reader additionally scans and rejects the 992 padded-row
    // positions (31 padded rows x 32 columns per tile) one element at a time. Native's own
    // force_row_major path collapses the padding to logical volume (an UntilizeWithUnpadding
    // bookend) before it ever transposes or scatters, so it pays the 32x cost nowhere. `rank == 1`
    // inputs normalize to a [1,1,1,N] working shape, whose row extent is 1 by construction. Index and
    // src necessarily share the unit row here: scatter requires index.shape[d] <= input.shape[d] for
    // every non-scatter axis d, and this row is never the scatter axis (the scatter axis is
    // transposed to last, so this is always the pre-last axis of the post-transpose shape) -- so a 1
    // here forces the same 1 on index/src without checking them separately.
    // A call the row-major detour serves never reaches the TILE factories, so it pays none of this:
    // its unit row is untilized to one logical stick before any kernel runs. The detour's own gate
    // decides that (prefers_row_major_strategy), on the same working shapes and output placement the
    // dispatch scatters, so a unit row the detour declines -- including one whose output placement
    // the detour cannot write -- is demoted rather than sent through the TILE factories.
    if (input_tensor.dtype() == DataType::BFLOAT16 && input_tensor.layout() == Layout::TILE) {
        const ttnn::Shape working_input = codegen_working_shape(input_tensor.logical_shape(), axis);
        const ttnn::Shape working_index = codegen_working_shape(index_tensor.logical_shape(), axis);
        if (working_input[-2] == 1 && !prefers_row_major_strategy(
                                          input_tensor,
                                          index_tensor,
                                          src_tensor,
                                          working_input,
                                          working_index,
                                          output_mem_config,
                                          /*transposed_copies_pending=*/axis != rank - 1)) {
            return true;
        }
    }

    return false;
}

}  // namespace ttnn::operations::data_movement::scatter
