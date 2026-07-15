// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Functional coverage for the packer-ReLU PackTile knob (PackRelu::Zero).
//
// Mode 0 (basic): copy in0 -> DEST -> pack with packer ReLU -> out. The packer clamps negatives to
// zero as it writes DEST -> L1, so out == relu(in0).
//
// Mode 1 (escape reset): a ReLU chain (packing to a dead sink CB) is followed by an INDEPENDENT
// linear chain that packs in1 -> out. STACC_RELU is a latched packer-global mode; the ReLU chain
// must restore pass-through at exit, otherwise the second (linear) pack would silently clamp in1's
// negatives. out must equal in1 unchanged.
//
// Mode 2 / Mode 3 (exp A/B): copy -> exp -> pack, WITH (2) and WITHOUT (3) packer ReLU. exp(x) > 0
// for all x, so ReLU is a no-op here; both variants must equal exp(in0). This proves the knob
// composes with an SFPU op and passes positive values through unchanged.
//
// Mode 4 (heterogeneous): ONE chain with two pack sites of DIFFERENT modes — relu (in0 -> c_16) and
// linear (in1 -> c_17). Each pack sets/restores its own mode per pack call, so out16 == relu(in0)
// and out17 == in1 (unchanged). Exercises the per-pack (non-homogeneous) ReLU path.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

void kernel_main() {
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_sink = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_out2 = tt::CBIndex::c_17;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t mode = get_compile_time_arg_val(1);  // 0 basic, 1 escape, 2 exp+relu, 3 exp plain, 4 mixed

    compute_kernel_hw_startup(cb_in0, cb_out);
    using namespace compute_kernel_lib;

    using ReluPack = PackTile<
        cb_out,
        OutputLifecycle::Streaming,
        PackTileReconfig::Output,
        Dst::D0,
        TileOffset::Unset,
        PackTileL1Accumulation::Disabled,
        PackRelu::Zero>;

    if constexpr (mode == 0) {
        eltwise_chain(
            EltwiseShape::tiles(n),
            CopyTile<cb_in0, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::Input>{},
            ReluPack{});
    } else if constexpr (mode == 1) {
        // Chain 1: relu pack to a dead sink — latches STACC_RELU, then restores it at chain exit.
        eltwise_chain(
            EltwiseShape::tiles(n),
            CopyTile<cb_in0, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::Input>{},
            PackTile<
                cb_sink,
                OutputLifecycle::Streaming,
                PackTileReconfig::Output,
                Dst::D0,
                TileOffset::Unset,
                PackTileL1Accumulation::Disabled,
                PackRelu::Zero>{});
        // Chain 2: independent LINEAR pack of in1 -> out. Correct only if chain 1 reset the packer.
        eltwise_chain(
            EltwiseShape::tiles(n),
            CopyTile<cb_in1, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::Input>{},
            PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::Output>{});
    } else if constexpr (mode == 2) {
        // copy -> exp -> pack WITH packer ReLU. exp > 0 so relu is a no-op; out == exp(in0).
        eltwise_chain(
            EltwiseShape::tiles(n),
            CopyTile<cb_in0, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::Input>{},
            Exp<>{},
            ReluPack{});
    } else if constexpr (mode == 3) {
        // copy -> exp -> pack WITHOUT packer ReLU (the A/B baseline for mode 2).
        eltwise_chain(
            EltwiseShape::tiles(n),
            CopyTile<cb_in0, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::Input>{},
            Exp<>{},
            PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::Output>{});
    } else {
        // Heterogeneous: relu pack (in0 -> c_16) and linear pack (in1 -> c_17) in ONE chain.
        eltwise_chain(
            EltwiseShape::tiles(n),
            CopyTile<cb_in0, Dst::D0, InputLifecycle::Streaming, CopyTileReconfig::Input>{},
            CopyTile<cb_in1, Dst::D1, InputLifecycle::Streaming, CopyTileReconfig::Input>{},
            PackTile<
                cb_out,
                OutputLifecycle::Streaming,
                PackTileReconfig::Output,
                Dst::D0,
                TileOffset::Unset,
                PackTileL1Accumulation::Disabled,
                PackRelu::Zero>{},
            PackTile<
                cb_out2,
                OutputLifecycle::Streaming,
                PackTileReconfig::Output,
                Dst::D1,
                TileOffset::Unset,
                PackTileL1Accumulation::Disabled,
                PackRelu::None>{});
    }
}
