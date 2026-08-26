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
    const ttnn::prim::PadCodegenParams& operation_attributes, const ttnn::prim::PadCodegenInputs& tensor_args) {
    const Tensor& input = tensor_args.input;
    if (input.layout() != Layout::ROW_MAJOR) {
        // supported_by_codegen() admits TILE only for whole-tile back-pads, so every transfer the
        // tiled reader issues is a whole tile page and it can never reach a staging fallback.
        return false;
    }

    // Native's RM pad-value packing has no FLOAT32 case and falls through to the BFLOAT16 one, so
    // it fills with a bf16-rounded constant for every nonzero float32 value (3.0 lands as
    // 3.00392, 65536.0 as 65679.0) while codegen is exact. Demoting these would buy speed with a
    // wrong answer, so they stay on codegen at whatever the staging path costs. Only float32 is
    // affected; bfloat16/int32/uint32 fills are byte-exact on both paths.
    if (input.dtype() == DataType::FLOAT32 && operation_attributes.packed_pad_value != 0) {
        return false;
    }

    // An input row that is not a multiple of the NOC transfer granularity cannot be read directly:
    // reader_pad_rm_interleaved.cpp has to pull the alignment-padded page into scratch and RISC-
    // memmove the real bytes out, which costs a barrier and a byte copy per stick and measures
    // 0.55-0.67x of native on Blackhole at every size tried up to 1.3MB. Every other shape reaches
    // that reader's batched fast path and wins 1.25-1.57x. The bound is the NOC granularity, so it
    // moves with the buffer: a width that stages against a 64B DRAM alignment can run direct
    // against a 32B one.
    //
    // tensor_args.input is always the 4D-unsqueezed tensor (PadCodegenInputs is only ever built
    // from input_4d), so logical_shape()[3] is W directly.
    const uint32_t stick_size = input.logical_shape()[3] * input.element_size();
    return stick_size % input.buffer()->alignment() != 0;
}

bool supported_execution_controls(bool use_multicore, const std::optional<CoreRangeSet>& sub_core_grids) {
    return use_multicore && !sub_core_grids.has_value();
}

}  // namespace ttnn::operations::data_movement::pad_codegen
