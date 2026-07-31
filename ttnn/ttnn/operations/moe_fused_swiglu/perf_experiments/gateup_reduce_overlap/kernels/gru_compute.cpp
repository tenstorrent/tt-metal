// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// gateup_reduce_overlap — ISOLATED PERF BENCH, COMPUTE kernel.
//
// Reconstructs moe_fused_swiglu's gate/up matmul + cross-column reduce (op_design.md §4.3,
// changelog.md Refinement 2 lever 2) for ONE grid column (KGROUPS=10 cores), so the ONLY thing
// under test is the SCHEDULE between the two stages:
//   BASELINE   (S=1): matmul the whole [m_eff, HN_PAD] gate/up block, THEN reduce the whole block
//              down the tree. This is the op's honest current approach (in1_num_subblocks==1,
//              GU_IN1_SUBBLOCKS == HN_PAD/HN_BLOCK == 1 today).
//   SPLIT_SERIAL   (S>1, PIPELINED=0): split into S sub-blocks (hidden axis) or S groups (M axis);
//              matmul stage s, reduce stage s, matmul stage s+1, reduce stage s+1, ... — same total
//              work as baseline, reordered, isolates the DEST-shrink cost alone (Refinement 2 lever
//              2's parked measurement, done honestly this time with NO overlap).
//   SPLIT_PIPELINED (S>1, PIPELINED=1): matmul stage s+1 is ISSUED before stage s's reduce-wait is
//              consumed, so the MATH/PACK engines have real work while the UNPACK thread's
//              cb_wait_front for stage s's reduce data is (hopefully) satisfied underneath it. This
//              is op_design.md §4.3's original, never-built design, generalised to two split axes
//              (SPLIT_AXIS=0 hidden / HN_BLOCK, SPLIT_AXIS=1 token-row / M_GROUP — the latter never
//              shrinks the matmul's out_subblock_w, so it should carry NO DEST cost).
//
// `add<>`/`copy<>`/`mul<>` (eltwise_convenience.hpp) take their CB ids as COMPILE-TIME
// InputSpec/OutputSpec template arguments, not runtime CircularBuffer refs — so "issue stage s+1
// before consuming stage s" needs an actual compile-time-unrolled schedule, not a runtime for-loop
// with a runtime stage index. StageRunner<STAGE, S, PIPELINED> supplies that unroll; S and PIPELINED
// are compile-time knobs (get_compile_time_arg_val), so this is ordinary template recursion, no raw
// LLK required.
//
// Topology: one grid COLUMN of KGROUPS=10 cores, EXACTLY moe_fused_swiglu's own reduce tree for
// column x=0 (root row0; children 1,2,4,8; row4's children 5,6; row6's child 7; row2's child 3;
// row8's child 9 — see program_descriptor.py's _reduce_tree, reproduced host-side here). No x/h
// multicast: within one column, x is NOT reuse-shared (each row owns a distinct K-slice), so this
// bench reads x/weights directly per core — the mcast machinery is a different, already-measured
// part of the op and is deliberately held trivial/absent here (concept isolation, /perf-lab).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

using namespace compute_kernel_lib;

constexpr uint32_t KR_PAD = get_compile_time_arg_val(0);      // uniform per-row K stride (23)
constexpr uint32_t HN_PAD = get_compile_time_arg_val(1);      // hidden tiles, this column (6 or 4)
constexpr uint32_t M_EFF = get_compile_time_arg_val(2);       // token tile-rows (8, 4, or 1)
constexpr uint32_t S = get_compile_time_arg_val(3);           // stage (sub-block) count
constexpr uint32_t SPLIT_AXIS = get_compile_time_arg_val(4);  // 0 = hidden (HN_BLOCK), 1 = M (M_GROUP)
constexpr uint32_t PIPELINED = get_compile_time_arg_val(5);   // 0 = split-serial, 1 = split-pipelined
constexpr uint32_t HN_BLOCK = get_compile_time_arg_val(6);
constexpr uint32_t M_GROUP = get_compile_time_arg_val(7);
constexpr uint32_t CB_X_BASE = get_compile_time_arg_val(8);
constexpr uint32_t CB_WG_BASE = get_compile_time_arg_val(9);
constexpr uint32_t CB_WU_BASE = get_compile_time_arg_val(10);
constexpr uint32_t CB_GATE_ACC_BASE = get_compile_time_arg_val(11);
constexpr uint32_t CB_UP_ACC_BASE = get_compile_time_arg_val(12);
// CB_A: root uses as gate-SiLU scratch; non-root uses as the gate "to-parent" send buffer.
// CB_B: root uses as the final SwiGLU (h_local) slot; non-root uses as the up "to-parent" send
// buffer. Root and non-root are mutually exclusive per core, so sharing keeps the CB count under
// the 64-CB limit at S=6/8 (see gru_program_descriptor.py).
constexpr uint32_t CB_A_BASE = get_compile_time_arg_val(13);
constexpr uint32_t CB_B_BASE = get_compile_time_arg_val(14);
constexpr uint32_t CB_REDUCE_GATE_BASE = get_compile_time_arg_val(15);
constexpr uint32_t CB_REDUCE_UP_BASE = get_compile_time_arg_val(16);

// Per-K-block FMA step count: KR_PAD is the uniform (padded) CB stride; `kr` (runtime, this row's
// REAL K extent — 22 or 23) bounds the loop so the pad tiles are never touched. Identical mechanism
// to the real op's KrSteps (moe_fused_swiglu_compute.cpp) — reconstructed locally, not included,
// so this experiment has zero header dependency on the real op.
struct KrSteps {
    uint32_t kr;
    ALWI uint32_t operator()(uint32_t, uint32_t) const { return kr; }
};

// Every stage has the SAME shape (only the CB index varies by stage): HN-split keeps all M_EFF rows
// per stage and narrows the hidden width to HN_BLOCK; M-split keeps the full HN_PAD width per stage
// and narrows the row count to M_GROUP. out_subblock_h stays 1 either way (pinned, matching the
// real op's OUT_SUBBLOCK_H_GU) — the M-split's DEST footprint (1 x HN_PAD) is therefore IDENTICAL to
// baseline's, which is exactly the option-4 hypothesis this bench separates from option-2/3's cost.
constexpr uint32_t STAGE_ROWS = (SPLIT_AXIS == 1) ? M_GROUP : M_EFF;
constexpr uint32_t STAGE_COLS = (SPLIT_AXIS == 0) ? HN_BLOCK : HN_PAD;
constexpr uint32_t STAGE_TILES = STAGE_ROWS * STAGE_COLS;

ALWI uint32_t x_cb(uint32_t stage) { return CB_X_BASE + ((SPLIT_AXIS == 1) ? stage : 0u); }
ALWI uint32_t wg_cb(uint32_t stage) { return CB_WG_BASE + ((SPLIT_AXIS == 0) ? stage : 0u); }
ALWI uint32_t wu_cb(uint32_t stage) { return CB_WU_BASE + ((SPLIT_AXIS == 0) ? stage : 0u); }

// ---- stage s's matmul: gate then up over the SAME resident x (WaitAndRetainOnLastBlock on both,
// exactly the real op's "cb_x_tiles consumed twice" contract), packed straight to out (num_k_blocks
// == 1, so no spill; gate_acc/up_acc double as the interm placeholder). ----
template <uint32_t STAGE>
ALWI void do_matmul(uint32_t kr) {
    MaybeDeviceZoneScope("compute_gateup");
    CircularBuffer xb(x_cb(STAGE));
    CircularBuffer wgb(wg_cb(STAGE));
    CircularBuffer wub(wu_cb(STAGE));
    CircularBuffer gacc(CB_GATE_ACC_BASE + STAGE);
    CircularBuffer uacc(CB_UP_ACC_BASE + STAGE);
    constexpr MatmulBlockShape shape = MatmulBlockShape::of(STAGE_ROWS, 1, 1, STAGE_COLS, KR_PAD, 1);

    matmul_block<
        /*transpose=*/false,
        /*packer_l1_acc=*/false,
        LastBlockTarget::Out,
        OutputCBLayout::SubblockMajor,
        matmul_config::InitMode::Short,
        InputPolicy::WaitAndRetainOnLastBlock,
        InputPolicy::WaitAndRetainOnLastBlock,
        NoPostCompute,
        NoPreKBlock,
        NoPostKBlock,
        /*untilize_block_ct_dim=*/0,
        KrSteps>(xb, wgb, gacc, gacc, shape, {}, {}, 0, 0, {}, KrSteps{kr});
    matmul_block<
        /*transpose=*/false,
        /*packer_l1_acc=*/false,
        LastBlockTarget::Out,
        OutputCBLayout::SubblockMajor,
        matmul_config::InitMode::Short,
        InputPolicy::WaitAndRetainOnLastBlock,
        InputPolicy::WaitAndRetainOnLastBlock,
        NoPostCompute,
        NoPreKBlock,
        NoPostKBlock,
        /*untilize_block_ct_dim=*/0,
        KrSteps>(xb, wub, uacc, uacc, shape, {}, {}, 0, 0, {}, KrSteps{kr});

    // x is shared across ALL HN-split stages (same cb id every stage) -> drain once, on the LAST
    // stage only. M-split's per-stage x is dedicated -> drain every stage. Weight is the mirror.
    if constexpr (SPLIT_AXIS == 1 || STAGE == S - 1) {
        xb.pop_front(STAGE_ROWS * KR_PAD);
    }
    if constexpr (SPLIT_AXIS == 0 || STAGE == S - 1) {
        wgb.pop_front(KR_PAD * STAGE_COLS);
        wub.pop_front(KR_PAD * STAGE_COLS);
    }
}

// ---- stage s's reduce: add every child's partial (non-root: plain `add`; root's LAST child:
// SiLU rides the packer via add_bias_bcast_rows), then either SwiGLU-multiply (root) or hand off
// to the writer for the unicast to this core's own parent (non-root). ----
template <uint32_t STAGE>
ALWI void do_reduce(uint32_t kr, uint32_t is_root, uint32_t num_children) {
    MaybeDeviceZoneScope("compute_reduce");
    CircularBuffer gacc(CB_GATE_ACC_BASE + STAGE);
    CircularBuffer uacc(CB_UP_ACC_BASE + STAGE);
    CircularBuffer rg(CB_REDUCE_GATE_BASE + STAGE);
    CircularBuffer ru(CB_REDUCE_UP_BASE + STAGE);
    CircularBuffer ab(CB_A_BASE + STAGE);
    CircularBuffer bb(CB_B_BASE + STAGE);

    for (uint32_t c = 0; c < num_children; ++c) {
        const bool final_child = (c + 1 == num_children);
        if (is_root && final_child) {
            // Root's last gate add: SiLU rides the PACKER thread (free, overlaps MATH).
            rg.wait_front(STAGE_TILES);
            for (uint32_t m = 0; m < STAGE_ROWS; ++m) {
                add_bias_bcast_rows<
                    BiasBroadcast::Elementwise,
                    OutputCBLayout::SubblockMajor,
                    bias_add_config::NoPostBias,
                    SiluActivation>(gacc, rg, ab, BiasAddShape::of(1, 1, 1, STAGE_COLS), {}, m * STAGE_COLS);
            }
            rg.pop_front(STAGE_TILES);
        } else {
            add<input(CB_GATE_ACC_BASE + STAGE), input(CB_REDUCE_GATE_BASE + STAGE), output(CB_GATE_ACC_BASE + STAGE)>(
                EltwiseShape::tiles(STAGE_TILES));
        }
        add<input(CB_UP_ACC_BASE + STAGE), input(CB_REDUCE_UP_BASE + STAGE), output(CB_UP_ACC_BASE + STAGE)>(
            EltwiseShape::tiles(STAGE_TILES));
    }

    if (is_root) {
        // FPU multiply through L1 (matches the real op's deliberate choice, examples/compute_fusion).
        mul<input(CB_A_BASE + STAGE), input(CB_UP_ACC_BASE + STAGE), output(CB_B_BASE + STAGE)>(
            EltwiseShape::tiles(STAGE_TILES));
    } else {
        copy<input(CB_GATE_ACC_BASE + STAGE), output(CB_A_BASE + STAGE)>(EltwiseShape::tiles(STAGE_TILES));
        copy<input(CB_UP_ACC_BASE + STAGE), output(CB_B_BASE + STAGE)>(EltwiseShape::tiles(STAGE_TILES));
    }
}

// Compile-time-unrolled schedule over the S stages. PIPELINED issues stage STAGE+1's matmul BEFORE
// consuming stage STAGE's reduce (so the math/pack engines have work while the reduce's
// cross-core wait is outstanding); the serial variant issues it after (same total work, no overlap
// attempt — isolates the DEST-shrink cost alone).
template <uint32_t STAGE, uint32_t NS, uint32_t DO_PIPELINE>
struct StageRunner {
    ALWI static void run(uint32_t kr, uint32_t is_root, uint32_t num_children) {
        if constexpr (DO_PIPELINE != 0 && STAGE + 1 < NS) {
            do_matmul<STAGE + 1>(kr);
        }
        do_reduce<STAGE>(kr, is_root, num_children);
        if constexpr (DO_PIPELINE == 0 && STAGE + 1 < NS) {
            do_matmul<STAGE + 1>(kr);
        }
        if constexpr (STAGE + 1 < NS) {
            StageRunner<STAGE + 1, NS, DO_PIPELINE>::run(kr, is_root, num_children);
        }
    }
};

void kernel_main() {
    const uint32_t kr = get_arg_val<uint32_t>(0);
    const uint32_t is_root = get_arg_val<uint32_t>(1);
    const uint32_t num_children = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup<SrcOrder::Reverse>(x_cb(0), wg_cb(0), CB_GATE_ACC_BASE);
    ActivationInitHelper<KernelActivation::SILU>::init();

    do_matmul<0>(kr);
    StageRunner<0, S, PIPELINED>::run(kr, is_root, num_children);
}
