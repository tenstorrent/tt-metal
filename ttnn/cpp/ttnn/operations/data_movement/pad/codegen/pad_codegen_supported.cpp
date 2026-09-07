// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_codegen_supported.hpp"

#include <tt-metalium/constants.hpp>

#include "pad_codegen_program_factory.hpp"

namespace ttnn::operations::data_movement::pad_codegen {

using namespace tt::tt_metal;
using namespace tt::constants;

bool supported_by_codegen(
    const ttnn::prim::PadCodegenParams& operation_attributes, const ttnn::prim::PadCodegenInputs& tensor_args) {
    const Tensor& input = tensor_args.input;
    const DataType dtype = input.dtype();

    // codegen_pad.py's invalidate_vector rejects bfloat8_b unconditionally, for BOTH layouts:
    // RM is a dtype-agnostic byte-copy kernel that explicitly excludes the block-float format
    // ("bfloat8_b not supported with ROW_MAJOR_LAYOUT"), and TILE pad-value fill corrupts
    // bf8_b's shared-per-tile exponent ("bfloat8_b TILE pad-value fill unsupported", observed
    // PCC -0.75 on silicon). uint16/bfloat4_b are accepted by ops/pad/pad.py's own dtype gate
    // but are outside codegen_pad.py's swept grid (manifest coverage note) -- left for a
    // follow-up broadening, not rejected as unsupported per se, but this port only covers what
    // was actually swept.
    if (dtype != DataType::BFLOAT16 && dtype != DataType::FLOAT32 && dtype != DataType::INT32 &&
        dtype != DataType::UINT32) {
        return false;
    }

    // manifest case (reason=left-out-for-now): ops/pad/pad.py unshards a sharded input (or a
    // requested sharded output) to interleaved DRAM before ever reaching a pad kernel -- citing
    // "catastrophic NOC contention" for a direct sharded RM pad. This port implements only the
    // two INTERLEAVED builders (build_pad_tiled / build_pad_rm); sharded placement stays native.
    if (input.memory_config().is_sharded() || operation_attributes.output_mem_config.is_sharded()) {
        return false;
    }

    const Layout layout = input.layout();
    if (layout == Layout::ROW_MAJOR) {
        // codegen_pad.py: the RM stick-copy kernel is dtype-agnostic byte movement -- front+back
        // padding on any dim, at any sub-tile amount, is representable.
        return true;
    }
    if (layout == Layout::TILE) {
        // codegen_pad.py: the TILE tile-page-copy kernel supports back-only padding (front-pad
        // on ANY dim, including N/C, is rejected -- _has_front_pad has no per-dim exception),
        // and every back-pad on H/W must be a whole number of tiles: a tile-page copy can only
        // place pad tiles at whole-tile boundaries, so a back-pad that leaves a tile straddling
        // real data and pad is unrepresentable by this kernel (manifest case, reason
        // real-kernel-limit).
        if (operation_attributes.front_n != 0 || operation_attributes.front_c != 0 ||
            operation_attributes.front_h != 0 || operation_attributes.front_w != 0) {
            return false;
        }
        const auto& in_shape = input.logical_shape();
        const uint32_t H = in_shape[2];
        const uint32_t W = in_shape[3];
        // The input's own last tile must be whole too. A tile-page copy moves a partial input tile
        // verbatim, remainder lanes and all, and nothing downstream of it writes the pad value into
        // those lanes -- ops/pad/spec.py reaches this path for a partial input only by first running
        // build_fill_partial_tile, which is not part of this port's kernel set.
        if (H % TILE_HEIGHT != 0 || W % TILE_WIDTH != 0) {
            return false;
        }
        const uint32_t back_h = operation_attributes.H_out - operation_attributes.front_h - H;
        const uint32_t back_w = operation_attributes.W_out - operation_attributes.front_w - W;
        return (back_h % TILE_HEIGHT == 0) && (back_w % TILE_WIDTH == 0);
    }
    return false;
}

bool is_demoted(
    const ttnn::prim::PadCodegenParams& /*operation_attributes*/, const ttnn::prim::PadCodegenInputs& /*tensor_args*/) {
    // No shape is perf-demoted. This gate used to demote row-major inputs whose stick size is not a
    // multiple of the buffer alignment, on the theory that reader_pad_rm_interleaved.cpp's staging
    // fallback -- pull the alignment-padded page into scratch, RISC-memmove the real bytes out --
    // costs a barrier and a byte copy per stick and so runs slower than native. Cross-arch CI does
    // not bear that out: over the ledger's 140 configs the 30 the predicate selected all beat native
    // on device on both arches, from 0.92x at the narrowest margin to 0.135x on blackhole and 0.091x
    // on wormhole_b0. Staging costs something, but native pays more elsewhere, so demoting these
    // gave away the port's largest wins. The gate stays as the routing extension point for a genuine
    // future device regression.
    return false;
}

bool supported_execution_controls(bool use_multicore, const std::optional<CoreRangeSet>& sub_core_grids) {
    return use_multicore && !sub_core_grids.has_value();
}

}  // namespace ttnn::operations::data_movement::pad_codegen
