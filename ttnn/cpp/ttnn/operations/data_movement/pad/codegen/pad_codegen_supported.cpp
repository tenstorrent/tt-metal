// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_codegen_supported.hpp"

#include <algorithm>
#include <initializer_list>

#include <tt-metalium/constants.hpp>

#include "pad_codegen_program_factory.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::constants;

bool supported_by_codegen(const PadCodegenParams& operation_attributes, const PadCodegenInputs& tensor_args) {
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
        const uint32_t back_h = operation_attributes.H_out - operation_attributes.front_h - H;
        const uint32_t back_w = operation_attributes.W_out - operation_attributes.front_w - W;
        return (back_h % TILE_HEIGHT == 0) && (back_w % TILE_WIDTH == 0);
    }
    return false;
}

bool is_demoted(const PadCodegenParams& operation_attributes, const PadCodegenInputs& tensor_args) {
    const Tensor& input = tensor_args.input;
    if (input.layout() != Layout::ROW_MAJOR) {
        // Every entry in the perf-demoted ledger is row_major; TILE never demotes.
        return false;
    }
    const DataType dtype = input.dtype();
    const auto& in_shape = input.logical_shape();
    const uint32_t H = in_shape[2];
    const uint32_t W = in_shape[3];
    const uint32_t front_h = operation_attributes.front_h;
    const uint32_t front_w = operation_attributes.front_w;
    const uint32_t H_out = operation_attributes.H_out;
    const uint32_t W_out = operation_attributes.W_out;

    auto shape_is = [&](uint32_t h, uint32_t w, uint32_t fh, uint32_t fw, uint32_t ho, uint32_t wo) {
        return H == h && W == w && front_h == fh && front_w == fw && H_out == ho && W_out == wo;
    };
    auto value_is = [&](float raw) { return operation_attributes.packed_pad_value == pack_pad_value(dtype, raw); };
    auto dtype_in = [&](std::initializer_list<DataType> set) {
        return std::find(set.begin(), set.end(), dtype) != set.end();
    };
    const std::initializer_list<DataType> kAllFourDtypes = {
        DataType::BFLOAT16, DataType::FLOAT32, DataType::INT32, DataType::UINT32};

    // [1, 1, 32, 32]|padding=[[0,0],[0,0],[0,32],[0,32]]&value=3|{bfloat16,int32}|row_major
    if (shape_is(32, 32, 0, 0, 64, 64) && value_is(3) && dtype_in({DataType::BFLOAT16, DataType::INT32})) {
        return true;
    }
    // [1, 1, 32, 32]|padding=[[0,0],[0,0],[3,25],[4,6]]&value=0|{bf16,fp32,int32,uint32}|row_major
    if (shape_is(32, 32, 3, 4, 60, 42) && value_is(0) && dtype_in(kAllFourDtypes)) {
        return true;
    }
    // [1, 1, 32, 32]|padding=[[0,0],[0,0],[3,25],[4,6]]&value=3|{fp32,int32,uint32}|row_major
    if (shape_is(32, 32, 3, 4, 60, 42) && value_is(3) &&
        dtype_in({DataType::FLOAT32, DataType::INT32, DataType::UINT32})) {
        return true;
    }
    // [1, 1, 64, 64]|padding=[[0,0],[0,0],[0,15],[0,31]]&value=0|{bf16,fp32,int32,uint32}|row_major
    if (shape_is(64, 64, 0, 0, 79, 95) && value_is(0) && dtype_in(kAllFourDtypes)) {
        return true;
    }
    // [1, 1, 64, 64]|padding=[[0,0],[0,0],[0,15],[0,31]]&value=3|{bf16,fp32,int32,uint32}|row_major
    if (shape_is(64, 64, 0, 0, 79, 95) && value_is(3) && dtype_in(kAllFourDtypes)) {
        return true;
    }
    // [1, 32, 32]|padding=[[0,0],[0,7],[0,9]]&value=0|{bf16,fp32,int32,uint32}|row_major
    if (shape_is(32, 32, 0, 0, 39, 41) && value_is(0) && dtype_in(kAllFourDtypes)) {
        return true;
    }
    // [1, 32, 32]|padding=[[0,0],[0,7],[0,9]]&value=3|{bf16,fp32,int32,uint32}|row_major
    if (shape_is(32, 32, 0, 0, 39, 41) && value_is(3) && dtype_in(kAllFourDtypes)) {
        return true;
    }
    // [32, 32]|padding=[[0,32],[0,32]]&value=0|{float32}|row_major
    if (shape_is(32, 32, 0, 0, 64, 64) && value_is(0) && dtype_in({DataType::FLOAT32})) {
        return true;
    }
    // [32, 32]|padding=[[4,2],[0,6]]&value=0|{bf16,fp32,int32,uint32}|row_major
    if (shape_is(32, 32, 4, 0, 38, 38) && value_is(0) && dtype_in(kAllFourDtypes)) {
        return true;
    }
    // [32, 32]|padding=[[4,2],[0,6]]&value=3|{bf16,fp32,int32,uint32}|row_major
    if (shape_is(32, 32, 4, 0, 38, 38) && value_is(3) && dtype_in(kAllFourDtypes)) {
        return true;
    }
    // [64, 64]|padding=[[0,31],[0,15]]&value=0|{bf16,fp32,int32,uint32}|row_major
    if (shape_is(64, 64, 0, 0, 95, 79) && value_is(0) && dtype_in(kAllFourDtypes)) {
        return true;
    }
    // [64, 64]|padding=[[0,31],[0,15]]&value=3|{bf16,fp32,int32,uint32}|row_major
    if (shape_is(64, 64, 0, 0, 95, 79) && value_is(3) && dtype_in(kAllFourDtypes)) {
        return true;
    }
    return false;
}

ImplementationSelector parse_implementation(std::string_view implementation) {
    if (implementation == "native") {
        return ImplementationSelector::Native;
    }
    if (implementation == "codegen") {
        return ImplementationSelector::Codegen;
    }
    return ImplementationSelector::Auto;
}

}  // namespace ttnn::prim
