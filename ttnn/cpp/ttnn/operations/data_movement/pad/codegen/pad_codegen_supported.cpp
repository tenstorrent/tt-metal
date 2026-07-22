// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_codegen_supported.hpp"

#include <algorithm>
#include <array>
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
    // tensor_args.input is always the 4D-unsqueezed tensor (try_pad_codegen builds
    // PadCodegenInputs from input_4d), so logical_shape()[0..3] is N,C,H,W directly -- matching
    // the ledger's shapes requires N/C too, not just H/W: several ledger entries (and, more
    // importantly, non-demoted sweep points outside the ledger) share the same H/W bucket
    // (H=32, W=32, front=0, out=64x64) but differ in C -- an H/W-only match would either miss the
    // C=3 ledger entry below or wrongly demote an unrelated C=1 config that happens to land in
    // the same bucket.
    const auto& in_shape = input.logical_shape();
    const uint32_t N = in_shape[0];
    const uint32_t C = in_shape[1];
    const uint32_t H = in_shape[2];
    const uint32_t W = in_shape[3];
    const auto& a = operation_attributes;

    auto shape_is = [&](uint32_t n,
                        uint32_t c,
                        uint32_t h,
                        uint32_t w,
                        uint32_t fn,
                        uint32_t fc,
                        uint32_t fh,
                        uint32_t fw,
                        uint32_t no,
                        uint32_t co,
                        uint32_t ho,
                        uint32_t wo) {
        return N == n && C == c && H == h && W == w && a.front_n == fn && a.front_c == fc && a.front_h == fh &&
               a.front_w == fw && a.N_out == no && a.C_out == co && a.H_out == ho && a.W_out == wo;
    };
    auto value_is = [&](float raw) { return a.packed_pad_value == pack_pad_value(dtype, raw); };
    auto dtype_in = [&](std::initializer_list<DataType> set) {
        return std::find(set.begin(), set.end(), dtype) != set.end();
    };
    const std::initializer_list<DataType> kAllFourDtypes = {
        DataType::BFLOAT16, DataType::FLOAT32, DataType::INT32, DataType::UINT32};
    // known_bad_golden (manifest): native's RM pad-value packing switch has no FLOAT32 case and
    // silently reuses BFLOAT16, corrupting any nonzero float32 fill -- these points are dropped
    // from correctness measurement entirely (no reliable native golden), so they never reach a
    // generic-vs-native device comparison and can never appear in the demoted ledger.
    const std::initializer_list<DataType> kNonzeroDemotableDtypes = {
        DataType::BFLOAT16, DataType::INT32, DataType::UINT32};

    // Four back-only-pad row_major shapes, each demoted at value=0 for all four dtypes and at
    // value=3 for all dtypes except float32 (see kNonzeroDemotableDtypes above). Transcribed
    // verbatim from the phase-7 perf-demoted ledger (measurements.demoted), not from any sweep --
    // see porting-guide.md's "Deriving is_demoted()".
    struct DemotedShape {
        uint32_t n, c, h, w, front_n, front_c, front_h, front_w, n_out, c_out, h_out, w_out;
    };
    constexpr std::array<DemotedShape, 4> kDemotedShapes{{
        // [1, 1, 64, 64]|padding=[[0,0],[0,0],[0,15],[0,31]]
        {1, 1, 64, 64, 0, 0, 0, 0, 1, 1, 79, 95},
        // [1, 32, 32]|padding=[[0,0],[0,7],[0,9]] (3D: unsqueeze_to_4D prepends N=1)
        {1, 1, 32, 32, 0, 0, 0, 0, 1, 1, 39, 41},
        // [32, 32]|padding=[[4,2],[0,6]] (2D: unsqueeze_to_4D prepends N=1, C=1)
        {1, 1, 32, 32, 0, 0, 4, 0, 1, 1, 38, 38},
        // [64, 64]|padding=[[0,31],[0,15]] (2D: unsqueeze_to_4D prepends N=1, C=1)
        {1, 1, 64, 64, 0, 0, 0, 0, 1, 1, 95, 79},
    }};

    for (const auto& s : kDemotedShapes) {
        if (!shape_is(
                s.n, s.c, s.h, s.w, s.front_n, s.front_c, s.front_h, s.front_w, s.n_out, s.c_out, s.h_out, s.w_out)) {
            continue;
        }
        if (value_is(0) && dtype_in(kAllFourDtypes)) {
            return true;
        }
        if (value_is(3) && dtype_in(kNonzeroDemotableDtypes)) {
            return true;
        }
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
