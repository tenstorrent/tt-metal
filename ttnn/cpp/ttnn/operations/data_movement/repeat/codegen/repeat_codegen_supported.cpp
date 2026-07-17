// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat/codegen/repeat_codegen_supported.hpp"

#include <algorithm>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/repeat/codegen/repeat_codegen_program_factory.hpp"

namespace ttnn::operations::data_movement::repeat_codegen {

namespace {

// Shared correctness rule for repeating dim `d` (in a tensor of rank `ndim`)
// by `reps` on `layout`/`dtype`, transcribed from codegen_repeat.py's
// invalidate_vector (both the base suite, which wraps the upstream
// sweeps.data_movement.repeat.repeat gate, and the broaden suite). A
// non-repeated dim (reps <= 1) never constrains the call.
//
// TILE: only H (d == ndim-2) or W (d == ndim-1) repeats require tile
// alignment on that axis. RepeatCodegen repeats in tile-page space
// (ceil(H/TILE_HEIGHT) / ceil(W/TILE_WIDTH) tiles); repeating a tile-page-
// quantized axis that isn't itself tile-aligned makes
// ceil(dim/tile)*reps disagree with ceil(dim*reps/tile), which
// shape-mismatches or folds pad rows into live data.
// Repeating N or C never touches this math -- the same tile page (padding
// included) is just duplicated, which is correct regardless of whether H/W
// happen to be sub-tile.
//
// RM last-dim (d == ndim-1) has an extra hardware constraint beyond
// correctness: RepeatCodegenProgramFactory's last-dim-RM branch sizes its
// output CB as kRepeatCbDepth pages of dst_buffer->aligned_page_size(), and
// that page IS the output stick -- `reps` copies of the input stick -- so the
// CB allocation grows linearly with reps and input width with no cap
// (mirrors ops/repeat/spec.py's "_CB_DEPTH ... repeat has no L1 clamp").
// Reject upfront any case whose projected CB wouldn't fit in one core's L1,
// so an oversized repeat cleanly routes to native (which streams the same
// output without a repeat-scaled CB) instead of TT_THROWing out of circular
// buffer allocation at program-compile time.
bool last_dim_rm_fits_in_l1(const Tensor& input, uint32_t reps) {
    if (input.storage_type() != tt::tt_metal::StorageType::DEVICE) {
        // Not yet on device (e.g. host-side probing); nothing to bound against.
        // Tensor::device() throws for a non-device tensor, so check first.
        return true;
    }
    const uint64_t out_stick_bytes =
        static_cast<uint64_t>(input.logical_shape()[-1]) * static_cast<uint64_t>(reps) * input.element_size();
    // The eventual output buffer type (DRAM vs L1) isn't known at this
    // routing-time check, so round up to the stricter (larger) of the two
    // alignments for a conservative (never-too-small) estimate.
    const auto& allocator = input.device()->allocator();
    const uint32_t dram_alignment = allocator->get_alignment(tt::tt_metal::BufferType::DRAM);
    const uint32_t l1_alignment = allocator->get_alignment(tt::tt_metal::BufferType::L1);
    const uint32_t alignment = std::max(dram_alignment, l1_alignment);
    const uint64_t out_aligned = tt::round_up(out_stick_bytes, static_cast<uint64_t>(alignment));
    const uint64_t projected_cb_bytes = out_aligned * ttnn::prim::kRepeatCbDepth;
    return projected_cb_bytes <= static_cast<uint64_t>(get_max_l1_space(input));
}

bool single_dim_ok(const Tensor& input, uint32_t d, uint32_t reps) {
    if (reps <= 1) {
        return true;
    }
    const auto& shape = input.logical_shape();
    const auto dtype = input.dtype();
    const auto layout = input.layout();
    const uint32_t ndim = shape.rank();
    if (layout == ttnn::ROW_MAJOR_LAYOUT) {
        if (dtype == tt::tt_metal::DataType::BFLOAT8_B) {
            return false;
        }
        if (dtype == tt::tt_metal::DataType::BFLOAT16 && shape[-1] < 2) {
            return false;
        }
        if (d == ndim - 1 && !last_dim_rm_fits_in_l1(input, reps)) {
            return false;
        }
        return true;
    }
    if (layout == ttnn::TILE_LAYOUT) {
        if (d == ndim - 1) {
            return shape[-1] % tt::constants::TILE_WIDTH == 0;
        }
        if (d == ndim - 2) {
            return shape[-2] % tt::constants::TILE_HEIGHT == 0;
        }
        return true;
    }
    return false;
}

}  // namespace

ImplementationSelector parse_implementation(const std::string& implementation) {
    if (implementation == "auto") {
        return ImplementationSelector::Auto;
    }
    if (implementation == "native") {
        return ImplementationSelector::Native;
    }
    if (implementation == "codegen") {
        return ImplementationSelector::Codegen;
    }
    TT_THROW("Unknown repeat implementation '{}': expected 'auto', 'native', or 'codegen'", implementation);
}

bool supported_by_codegen(const Tensor& input, uint32_t rep_dim, uint32_t num_repeats) {
    // Sharded input: RepeatCodegen only ever reaches sharded tensors via a
    // native unshard-to-interleaved-DRAM hop that this port does not
    // implement; see the manifest's hand-authored sharded case.
    if (input.memory_config().is_sharded()) {
        return false;
    }
    const auto& shape = input.logical_shape();
    if (shape.rank() != 4 || rep_dim >= 4) {
        return false;
    }
    return single_dim_ok(input, rep_dim, num_repeats);
}

bool supported_by_codegen(const Tensor& input, const ttsl::SmallVector<uint32_t>& repeat_dims) {
    if (input.memory_config().is_sharded()) {
        return false;
    }
    const auto& shape = input.logical_shape();
    const uint32_t ndim = shape.rank();
    // repeat_codegen's kernels assume a 4D-padded tensor (ops/repeat/spec.py's
    // _page_map / build_repeat_rm_factory); a rank > 4 input has no path here.
    if (repeat_dims.size() != ndim || ndim < 2 || ndim > 4) {
        return false;
    }
    bool any_repeated = std::any_of(repeat_dims.cbegin(), repeat_dims.cend(), [](uint32_t r) { return r > 1; });
    if (!any_repeated) {
        return false;
    }
    for (uint32_t d = 0; d < ndim; ++d) {
        if (!single_dim_ok(input, d, repeat_dims[d])) {
            return false;
        }
    }
    return true;
}

bool is_demoted(const Tensor& /*input*/, const ttsl::SmallVector<uint32_t>& /*repeat_dims*/) {
    // No shape is perf-demoted. On device the ported path holds parity with
    // generic_op and beats native everywhere measured: row-major ~1.9x, and
    // tile H-broadcast up to ~5-12x over native's untilize/repeat/tilize
    // composite. Wall-clock sits at parity with native within the host-dispatch
    // jitter floor on these sub-microsecond-kernel shapes. The gate stays as the
    // routing extension point for a genuine future device regression.
    return false;
}

}  // namespace ttnn::operations::data_movement::repeat_codegen
