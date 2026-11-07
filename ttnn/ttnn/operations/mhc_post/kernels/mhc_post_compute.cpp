// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post compute — plain-2d and stats-2d-tree regimes. Every phase is an eltwise_chain or
// matmul_block helper call; the one non-library piece is the SfpuRowSum chain element below
// (a thin UnaryOp wrapper over sfpu_reduce, see "Helper gaps").
//
// mix_block (all modes), per (j, c) output tile of a block (IterationShape::grid(n, extent)):
//   D1 <- bcast_col(post_j)          D2 <- F_c             D0 = D1 * D2          (MulBinary)
//   for i: D1 <- bcast_col(comb_ij)  D2 <- X_{i,c}         D0 = D0 + 1.0*D1*D2   (Addcmul, fused MAD)
//   pack D0 -> cb_out_tiles (per tile)
//   fused: pack D0 -> cb_xprime_mm (same format as X' -> same bits),
//          D3 <- ss_run; D3 = D3 + 1.0*D0*D0 (unrounded D0, exact fp32); pack D3 -> ss_run
// The coefficient tiles carry their value in column 0 (the reader's expansion), so every
// consumer is a Col broadcast; post_j sits at tile j, comb[i][j] at n + i*n + j (comb is applied
// transposed). F / X / coefficients are caller-managed (None/None).
//
// mix_fused (Refinement 5; Blackhole, bf16 X streams — host gate): pack_coefficients compacts the
//   row's broadcast coefficients into packed DEST-layout tiles once per row, and mix_block_fused /
//   collapse_block_fused run the same arithmetic as 3 (mix) / 2 (y) multi-term SFPU passes per output
//   tile (mhc_post_mix_sfpu.hpp). Bit-identical to the per-term chains above, which remain the
//   mix_fused = 0 path.
// project_block (fused): matmul_block of the bf16 X' block (K = n*extent tiles, order [j][c])
//   against the resident W slice (block-major, same order) -> cb_proj_block (fp32), then an
//   exact SFPU add into the running row partial (cb_proj_run; the row's last block packs
//   straight into cb_partial for the writer).
//   Exact variant (HiFi3/HiFi4, proj_passes = 16): the FPU rounds every 16-term face partial sum
//   to ~tf32, so the block is projected in 16 passes, pass j on X' masked to columns {j, j+16}
//   (one product per face -> exact in fp32 DEST), each accumulated with the exact SFPU add.
// finish_row_partial (fused): ss_run -> SfpuRowSum -> cb_ss_row (col 0); the writer patches it
//   into column 24 of the partial and ships it to the group leader.
// combine_row (fused): sum the gathered partials in slot-index order (exact SFPU adds, running
//   sum resident in DEST, slots streamed as they arrive) -> cb_lead_out (two-level) / cb_stats_row.
// collapse_block (collapse): y_c = sum_i P_i * X'_{i,c} on the SFPU in fp32 from the bf16 X' block
//   already in L1 (cb_xprime_mm, before the projection pops it), packed once to bf16 (cb_y_tiles).
// hc_row (collapse, on the core that produced the row's stats): the writer permutes the stats row
//   into hc lanes; HcProgram (mhc_post_hc_sfpu.hpp, a custom lane-wise fp32 SFPU program inside an
//   eltwise_chain) runs the hc step in place; the result lanes go back to the writer.
//
// Helper gaps (recorded, not bypassed): no kernel_lib element wraps the SFPU row reduce, so
// SfpuRowSum is a 5-line UnaryOp element (init/exec only — the chain still owns DEST sync, CB
// lifecycle and reconfig). The FPU reduce helper is rejected: it reads fp32 through srcA/B (tf32),
// ~1e-3 rel error vs the 1e-6 SUMSQ gate. The hc step is a custom SFPU program (HcProgram): no
// whole-tile helper can sum across quantities that sit in different tile rows (op_design.md →
// "Helpers considered and rejected"); it is wrapped as a chain element, so the chain still owns
// DEST sync, CB lifecycle and reconfig.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/ternary/ternary.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/ttnn/operations/mhc_post/kernels/mhc_post_combine.hpp"
#include "ttnn/ttnn/operations/mhc_post/kernels/mhc_post_hc_sfpu.hpp"
#include "ttnn/ttnn/operations/mhc_post/kernels/mhc_post_mix_sfpu.hpp"

namespace ckl = compute_kernel_lib;

namespace {

constexpr uint32_t cb_coef_cols = get_compile_time_arg_val(0);
constexpr uint32_t cb_f_tiles = get_compile_time_arg_val(1);
constexpr uint32_t cb_x_tiles = get_compile_time_arg_val(2);
constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(3);
constexpr uint32_t n = get_compile_time_arg_val(4);
constexpr uint32_t block_c = get_compile_time_arg_val(5);
constexpr uint32_t n_coef_cols = get_compile_time_arg_val(6);
// fused block (see mhc_post_program_descriptor.py fused_ct)
constexpr uint32_t FB = 7;
constexpr bool fused = get_compile_time_arg_val(FB + 0) != 0;
constexpr uint32_t cb_xprime_mm = get_compile_time_arg_val(FB + 1);
constexpr uint32_t cb_w_slice = get_compile_time_arg_val(FB + 2);
constexpr uint32_t cb_proj_block = get_compile_time_arg_val(FB + 3);
constexpr uint32_t cb_proj_run = get_compile_time_arg_val(FB + 4);
constexpr uint32_t cb_ss_run = get_compile_time_arg_val(FB + 5);
constexpr uint32_t cb_ss_row = get_compile_time_arg_val(FB + 6);
constexpr uint32_t cb_partial = get_compile_time_arg_val(FB + 7);
constexpr uint32_t cb_gather = get_compile_time_arg_val(FB + 8);
constexpr uint32_t cb_root_gather = get_compile_time_arg_val(FB + 9);
constexpr uint32_t cb_lead_out = get_compile_time_arg_val(FB + 10);
constexpr uint32_t cb_stats_row = get_compile_time_arg_val(FB + 12);
constexpr uint32_t Q = get_compile_time_arg_val(FB + 13);
constexpr uint32_t COMBINE_FANIN = get_compile_time_arg_val(FB + 14);
constexpr uint32_t COMBINE_LAG = get_compile_time_arg_val(FB + 16);
constexpr uint32_t proj_passes = get_compile_time_arg_val(FB + 22);
constexpr uint32_t cb_proj_masks = get_compile_time_arg_val(FB + 23);
constexpr uint32_t cb_xprime_masked = get_compile_time_arg_val(FB + 24);
constexpr bool collapse = get_compile_time_arg_val(FB + 25) != 0;
constexpr uint32_t cb_y_tiles = get_compile_time_arg_val(FB + 26);
constexpr uint32_t cb_hc_lanes = get_compile_time_arg_val(FB + 27);
constexpr uint32_t cb_hc_out = get_compile_time_arg_val(FB + 28);
constexpr uint32_t HC_LAG = get_compile_time_arg_val(FB + 30);  // after fused_ct (descriptor HC_LAG)
static_assert(HC_LAG <= 1, "hc runs at most one iteration after its combine");
// Fused SFPU mix (Refinement 5, mhc_post_mix_sfpu.hpp): packed coefficient tiles + multi-term SFPU
// passes. 0 = the per-term chain below (any arch).
constexpr bool mix_fused = get_compile_time_arg_val(FB + 31) != 0;
constexpr uint32_t cb_coef_packed = get_compile_time_arg_val(FB + 32);
// F's dtype differs from X's (e.g. fp32 sublayer output into bf16 streams): F and X then sit in
// different unpack classes (fp32 unpack-to-dest vs bf16 srcA), so the raw mix loop re-inits for X.
constexpr bool f_differs_x = get_compile_time_arg_val(FB + 33) != 0;
// Perf 2: the mix's c-pair window (mix_block_fused). 0 = single-tile windows only (host carve-out).
constexpr bool mix_pairs = get_compile_time_arg_val(FB + 34) != 0;

static_assert(n == 4, "mhc_post: the mix chain is written for n = 4 streams");

constexpr uint32_t ONE_F32_BITS = 0x3f800000u;  // Addcmul value = 1.0 (exact)
constexpr uint32_t COEF_POST_BASE = 0;
constexpr uint32_t COEF_COMB_BASE = n;
constexpr uint32_t COEF_PRE_BASE = n + n * n;  // collapse: P_i at n + n^2 + i
// Packed coefficient tiles (mix_fused): A_j = {post_j, comb_0j, comb_1j, comb_2j} at j, P at n.
constexpr uint32_t PACKED_A_BASE = 0;
constexpr uint32_t PACKED_P = n;
// B_j = comb_3j as a full (lane-aligned) broadcast fp32 tile, copied (not broadcast-unpacked) per
// output tile by the raw mix loop.
constexpr uint32_t PACKED_B_BASE = n + (collapse ? 1 : 0);
constexpr uint32_t n_packed = n + (collapse ? 1 : 0) + n;
static_assert(mhc_post_mix::COEFS_PER_PACKED == n, "A_j / P packing assumes n = 4 coefficients per tile");

// ---------------------------------------------------------------------------
// Chain specs
// ---------------------------------------------------------------------------
constexpr auto coef_in = ckl::input(
    cb_coef_cols,
    ckl::WaitPolicy::None,
    ckl::PopPolicy::None,
    ckl::InputTileMapping::Col,
    ckl::DataFormatReconfig::Enabled,
    ckl::TileAddressing::Offset);
constexpr auto f_in = ckl::input(
    cb_f_tiles,
    ckl::WaitPolicy::None,
    ckl::PopPolicy::None,
    ckl::InputTileMapping::Row,
    ckl::DataFormatReconfig::Enabled,
    ckl::TileAddressing::Direct);
constexpr auto x_in = ckl::input(
    cb_x_tiles,
    ckl::WaitPolicy::None,
    ckl::PopPolicy::None,
    ckl::InputTileMapping::Row,
    ckl::DataFormatReconfig::Enabled,
    ckl::TileAddressing::Offset);
constexpr auto out_spec = ckl::output(cb_out_tiles, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile);

using CoefBcast = ckl::UnaryBcast<ckl::BroadcastDim::Col, coef_in, ckl::Dst::D1>;
using CopyF = ckl::CopyTile<f_in, ckl::Dst::D2>;
using CopyX = ckl::CopyTile<x_in, ckl::Dst::D2>;
using MulPostF = ckl::MulBinary<ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D0>;
using MacCombX = ckl::Addcmul<DataFormat::Float32, ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D0>;
using PackOut = ckl::PackTile<out_spec, ckl::Dst::D0>;

// Fused mix extension: second X' pack (matmul / y operand) and the exact ss accumulation.
using PackXprime = ckl::PackTile<ckl::output(cb_xprime_mm), ckl::Dst::D0>;
using LoadSs = ckl::CopyTile<ckl::input(cb_ss_run), ckl::Dst::D3>;
using MacSs = ckl::Addcmul<DataFormat::Float32, ckl::Dst::D3, ckl::Dst::D0, ckl::Dst::D0, ckl::Dst::D3>;
using PackSs = ckl::PackTile<ckl::output(cb_ss_run), ckl::Dst::D3>;

// Exact fp32 row sum on the SFPU (result in column 0 of each row).
template <ckl::Dst Slot>
struct SfpuRowSum : ckl::UnaryOp<SfpuRowSum<Slot>, Slot> {
    static ALWI void init() { sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(ckl::to_u32(Slot) + slot_offset);
    }
};

// Pinned (Scalar, caller-managed) read of tile `base` of a staged CB.
template <uint32_t cb, ckl::Dst Slot>
using CopyAt = ckl::CopyTile<
    ckl::input(
        cb,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Scalar,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset),
    Slot>;
// Streaming (per tile wait/pop) read of a running tile.
template <uint32_t cb, ckl::Dst Slot>
using CopyRun = ckl::CopyTile<ckl::input(cb), Slot>;
template <uint32_t cb>
using PackTo = ckl::PackTile<ckl::output(cb), ckl::Dst::D0>;
using AddD0D1 = ckl::AddBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>;
// Exact projection mask pass: X' block tile i (srcA, 8 significant bits) * column mask j (srcB).
using MaskX = ckl::BinaryFpu<
    ckl::BinaryFpuOp::Mul,
    ckl::input(
        cb_xprime_mm,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Block,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset),
    ckl::input(
        cb_proj_masks,
        ckl::BroadcastDim::None,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Scalar,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset),
    ckl::Dst::D0>;

// collapse_block: X'_{i,c} (bf16, exact through srcA). The X' block is packed compact [j][c]
// (stride = the block's extent, as the projection reads it), so X'_i sits at Offset i*extent.
using CopyXprime = ckl::CopyTile<
    ckl::input(
        cb_xprime_mm,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Row,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset),
    ckl::Dst::D2>;
using PackY =
    ckl::PackTile<ckl::output(cb_y_tiles, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile), ckl::Dst::D0>;
// hc lanes in / out (caller-managed windows of LANE_TILES pages).
template <ckl::Dst Slot>
using LoadLanes = CopyAt<cb_hc_lanes, Slot>;
template <ckl::Dst Slot>
using PackLanes = ckl::PackTile<
    ckl::output(
        cb_hc_out,
        ckl::ReservePolicy::None,
        ckl::PushPolicy::None,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset),
    Slot>;

// Fused mix (mix_fused) elements.
template <ckl::Dst Slot>
using CoefBcastTo = ckl::UnaryBcast<ckl::BroadcastDim::Col, coef_in, Slot>;  // tile base + row
template <ckl::Dst Slot>
using CoefBcastAt = ckl::UnaryBcast<
    ckl::BroadcastDim::Col,
    ckl::input(
        cb_coef_cols,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Scalar,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset),
    Slot>;
using PackPacked = ckl::PackTile<
    ckl::output(
        cb_coef_packed,
        ckl::ReservePolicy::None,
        ckl::PushPolicy::None,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset),
    ckl::Dst::D0>;
using CopyPackedAt = CopyAt<cb_coef_packed, ckl::Dst::D3>;
template <ckl::Dst Slot>
using CopyXprimeTo = ckl::CopyTile<
    ckl::input(
        cb_xprime_mm,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Row,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset),
    Slot>;

// ---------------------------------------------------------------------------
// Block operations
// ---------------------------------------------------------------------------

// pack_coefficients (mix_fused, once per row): the row's column-broadcast coefficients -> packed
// tiles (mhc_post_mix_sfpu.hpp). A_j gathers post_j, comb_0j, comb_1j, comb_2j (tile bases 0, n,
// 2n, 3n + row j of a grid(n, 1) walk); collapse adds P = {P_0..P_3}. B_j = comb_3j as a full
// broadcast tile (read lane-aligned by mhc_mix_pass3). The caller owns the packed window.
ALWI void pack_coefficients(DataflowBuffer& coef, DataflowBuffer& packed) {
    ckl::eltwise_chain(
        ckl::IterationShape::grid(n, 1),
        CoefBcastTo<ckl::Dst::D0>{coef, COEF_POST_BASE},
        CoefBcastTo<ckl::Dst::D1>{coef, COEF_COMB_BASE + 0 * n},
        CoefBcastTo<ckl::Dst::D2>{coef, COEF_COMB_BASE + 1 * n},
        CoefBcastTo<ckl::Dst::D3>{coef, COEF_COMB_BASE + 2 * n},
        mhc_post_mix::CoefCompact{},
        PackPacked{packed, PACKED_A_BASE});
    if constexpr (collapse) {
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            CoefBcastAt<ckl::Dst::D0>{coef, COEF_PRE_BASE + 0},
            CoefBcastAt<ckl::Dst::D1>{coef, COEF_PRE_BASE + 1},
            CoefBcastAt<ckl::Dst::D2>{coef, COEF_PRE_BASE + 2},
            CoefBcastAt<ckl::Dst::D3>{coef, COEF_PRE_BASE + 3},
            mhc_post_mix::CoefCompact{},
            PackPacked{packed, PACKED_P});
    }
    ckl::eltwise_chain(
        ckl::IterationShape::grid(n, 1),
        CoefBcastTo<ckl::Dst::D0>{coef, COEF_COMB_BASE + 3 * n},
        PackPacked{packed, PACKED_B_BASE});
}

// mix_block, fused SFPU form (mix_fused). Same operands / SFPMUL / SFPMAD sequence as the per-term
// chain (bit-identical X' / ss), push order j-major, c-minor.
//
// Raw compute-API loop, not an eltwise_chain (helper bypass, Perf 1): the chain is non-uniform here
// (fp32 UnpackToDest operands, bf16 srcA operands, 3 SFPU programs), so it re-emits every element's
// init (8 copy / bcast inits, 3 SFPU inits, 4 srcA reconfigs) on every output tile. This loop emits a
// copy init only when the srcA operand class switches (fp32 unpack-to-dest <-> bf16 srcA) and the
// SFPU init once per block (the datacopy init only touches ADDR_MOD_0..3; the SFPU programs use
// ADDR_MOD_6/7). Measured (perf_experiments/mix_overhead v3/v7): focus collapse 107.5 -> 97.5 us,
// bit-identical. Out of line + Os, and the copies out of line, to keep the profiler build inside the
// kernel-config text budget (same speed as the inlined form).
//
// Perf 2 (perf_experiments/mix_reuse v2): c-pair window. Each fp32 unpack-to-dest copy (A_j, B_j,
// ss) is a full unpack <-> math serialization point (~190 cycles), the costliest thing in the loop
// (ablation: removing them -9.1 us of a 31 us compute_mix). The pair window shares A_j, B_j and the
// ss round trip across two c-tiles: D0 / D1 accumulate X'_{j,c} / X'_{j,c+1}, D2 holds the data
// operand (reloaded per term per c), D3 holds A_j, then B_j, then ss — 13 copies + 5 packs per 2
// output tiles instead of 16 + 6. Odd tail: the single-tile window (mix_tile). Measured, bit-identical:
// compute_mix 31.0 -> 25.4 us (focus); whole op plain T4096 C1792 -6.3 %, plain T640 C7168 -5.7 %,
// collapse T160 HiFi4 -4.7 %. Carve-out (mix_pairs = 0): collapse with the single-pass projection,
// measured slower (DRAM-bound body) — see _mix_pair_window in the descriptor. No chain can hold two
// output accumulators across terms in one DEST acquire, or share a coefficient / ss tile across c.
__attribute__((noinline)) void raw_copy(uint32_t cb, uint32_t idx, uint32_t dst) { copy_tile(cb, idx, dst); }
__attribute__((noinline)) void raw_copy_init(uint32_t cb) { copy_init(cb); }

#define MHC_SFPU(...) MATH((_llk_math_eltwise_unary_sfpu_params_(__VA_ARGS__, 0, VectorMode::RC_custom)))

// One output tile, the single-tile window (D3 = A_j, D0 = F, D1 / D2 = X; 3 SFPU passes).
template <bool with_ss>
__attribute__((noinline, optimize("Os"))) void mix_tile(
    uint32_t j, uint32_t c, DataflowBuffer& out, DataflowBuffer& xprime, DataflowBuffer& ss) {
    tile_regs_acquire();
    raw_copy(cb_coef_packed, PACKED_A_BASE + j, 3);  // fp32: state carried from the last window
    reconfig_data_format_srca(cb_coef_packed, cb_f_tiles);
    raw_copy_init(cb_f_tiles);
    raw_copy(cb_f_tiles, c, 0);
    if constexpr (f_differs_x) {
        reconfig_data_format_srca(cb_f_tiles, cb_x_tiles);
        raw_copy_init(cb_x_tiles);
    }
    raw_copy(cb_x_tiles, 0 * block_c + c, 1);
    raw_copy(cb_x_tiles, 1 * block_c + c, 2);
    MHC_SFPU(::ckernel::sfpu::mhc_mix_pass1);
    raw_copy(cb_x_tiles, 2 * block_c + c, 1);
    MHC_SFPU(::ckernel::sfpu::mhc_mix_pass2);
    raw_copy(cb_x_tiles, 3 * block_c + c, 1);
    reconfig_data_format_srca(cb_x_tiles, cb_coef_packed);
    raw_copy_init(cb_coef_packed);
    raw_copy(cb_coef_packed, PACKED_B_BASE + j, 3);
    if constexpr (with_ss) {
        ss.wait_front(1);
        raw_copy(cb_ss_run, 0, 2);
        ss.pop_front(1);
        MHC_SFPU(::ckernel::sfpu::mhc_mix_pass3<true>);
    } else {
        MHC_SFPU(::ckernel::sfpu::mhc_mix_pass3<false>);
    }
    tile_regs_commit();
    tile_regs_wait();
    out.reserve_back(1);
    pack_tile(0, cb_out_tiles);
    out.push_back(1);
    if constexpr (with_ss) {
        xprime.reserve_back(1);
        pack_tile(0, cb_xprime_mm);
        xprime.push_back(1);
        ss.reserve_back(1);
        pack_reconfig_data_format(cb_xprime_mm, cb_ss_run);
        pack_tile(2, cb_ss_run);
        pack_reconfig_data_format(cb_ss_run, cb_out_tiles);
        ss.push_back(1);
    }
    tile_regs_release();
}

#ifdef ARCH_BLACKHOLE
// Pair window term: D[c] += A_j[QC] * X_i (data operand reloaded into D2 per c).
template <uint32_t QC>
ALWI void mix_pair_term(uint32_t xi) {
    raw_copy(cb_x_tiles, xi, 2);
    MHC_SFPU(::ckernel::sfpu::mhc_r_mac<0, QC>);
    raw_copy(cb_x_tiles, xi + 1, 2);
    MHC_SFPU(::ckernel::sfpu::mhc_r_mac<1, QC>);
}

// Two output tiles X'_{j,c}, X'_{j,c+1} in one DEST acquire.
template <bool with_ss>
__attribute__((noinline, optimize("Os"))) void mix_pair(
    uint32_t j, uint32_t c, DataflowBuffer& out, DataflowBuffer& xprime, DataflowBuffer& ss) {
    tile_regs_acquire();
    raw_copy(cb_coef_packed, PACKED_A_BASE + j, 3);  // fp32 class (carried from the last window)
    reconfig_data_format_srca(cb_coef_packed, cb_f_tiles);
    raw_copy_init(cb_f_tiles);
    raw_copy(cb_f_tiles, c, 0);
    raw_copy(cb_f_tiles, c + 1, 1);
    if constexpr (f_differs_x) {
        reconfig_data_format_srca(cb_f_tiles, cb_x_tiles);
        raw_copy_init(cb_x_tiles);
    }
    MHC_SFPU(::ckernel::sfpu::mhc_r_mul<true>);
    mix_pair_term<mhc_post_mix::Q1_>(0 * block_c + c);
    mix_pair_term<mhc_post_mix::Q2_>(1 * block_c + c);
    mix_pair_term<mhc_post_mix::Q3_>(2 * block_c + c);
    reconfig_data_format_srca(cb_x_tiles, cb_coef_packed);
    raw_copy_init(cb_coef_packed);
    raw_copy(cb_coef_packed, PACKED_B_BASE + j, 3);
    reconfig_data_format_srca(cb_coef_packed, cb_x_tiles);
    raw_copy_init(cb_x_tiles);
    raw_copy(cb_x_tiles, 3 * block_c + c, 2);
    MHC_SFPU(::ckernel::sfpu::mhc_r_lane<0>);
    raw_copy(cb_x_tiles, 3 * block_c + c + 1, 2);
    MHC_SFPU(::ckernel::sfpu::mhc_r_lane<1>);
    reconfig_data_format_srca(cb_x_tiles, cb_coef_packed);
    raw_copy_init(cb_coef_packed);
    if constexpr (with_ss) {
        ss.wait_front(1);
        raw_copy(cb_ss_run, 0, 3);
        ss.pop_front(1);
        MHC_SFPU(::ckernel::sfpu::mhc_r_sq<true>);
    }
    tile_regs_commit();
    tile_regs_wait();
    out.reserve_back(2);
    pack_tile(0, cb_out_tiles);
    pack_tile(1, cb_out_tiles);
    out.push_back(2);
    if constexpr (with_ss) {
        xprime.reserve_back(2);
        pack_tile(0, cb_xprime_mm);
        pack_tile(1, cb_xprime_mm);
        xprime.push_back(2);
        ss.reserve_back(1);
        pack_reconfig_data_format(cb_xprime_mm, cb_ss_run);
        pack_tile(3, cb_ss_run);
        pack_reconfig_data_format(cb_ss_run, cb_out_tiles);
        ss.push_back(1);
    }
    tile_regs_release();
}
#endif

template <bool with_ss>
void mix_block_fused(uint32_t extent, DataflowBuffer& out, DataflowBuffer& xprime, DataflowBuffer& ss) {
    reconfig_data_format_srca(cb_coef_packed);
    raw_copy_init(cb_coef_packed);
    pack_reconfig_data_format(cb_out_tiles);
    MATH((::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::unused, DST_ACCUM_MODE>()));
    for (uint32_t j = 0; j < n; ++j) {
        uint32_t c = 0;
#ifdef ARCH_BLACKHOLE
        if constexpr (mix_pairs) {
#pragma GCC unroll 0
            for (; c + 1 < extent; c += 2) {
                mix_pair<with_ss>(j, c, out, xprime, ss);
            }
        }
#endif
#pragma GCC unroll 0
        for (; c < extent; ++c) {
            mix_tile<with_ss>(j, c, out, xprime, ss);
        }
    }
}

// collapse_block, fused SFPU form (mix_fused): 5 unpacks + 2 SFPU passes per y tile.
ALWI void collapse_block_fused(uint32_t extent, DataflowBuffer& packed, DataflowBuffer& xprime, DataflowBuffer& y) {
    ckl::eltwise_chain(
        ckl::IterationShape::grid(1, extent),
        CopyPackedAt{packed, PACKED_P},
        CopyXprimeTo<ckl::Dst::D0>{xprime, 0 * extent},
        CopyXprimeTo<ckl::Dst::D1>{xprime, 1 * extent},
        mhc_post_mix::YPass1{},
        CopyXprimeTo<ckl::Dst::D1>{xprime, 2 * extent},
        CopyXprimeTo<ckl::Dst::D2>{xprime, 3 * extent},
        mhc_post_mix::YPass2{},
        PackY{y});
}

// collapse_block: y = P_0 X'_0 + sum_{i>0} P_i X'_i over the block's `extent` c-tiles (fp32 in DEST,
// one bf16 rounding at the pack). Caller has waited the X' block.
ALWI void collapse_block(uint32_t extent, DataflowBuffer& coef, DataflowBuffer& xprime, DataflowBuffer& y) {
    ckl::eltwise_chain(
        ckl::IterationShape::grid(1, extent),
        CoefBcast{coef, COEF_PRE_BASE + 0},
        CopyXprime{xprime, 0 * extent},
        MulPostF{},
        CoefBcast{coef, COEF_PRE_BASE + 1},
        CopyXprime{xprime, 1 * extent},
        MacCombX{ONE_F32_BITS},
        CoefBcast{coef, COEF_PRE_BASE + 2},
        CopyXprime{xprime, 2 * extent},
        MacCombX{ONE_F32_BITS},
        CoefBcast{coef, COEF_PRE_BASE + 3},
        CopyXprime{xprime, 3 * extent},
        MacCombX{ONE_F32_BITS},
        PackY{y});
}

// NATURAL hc layout: exact 32-bit in-place transpose of each 16x16 face of DEST tile 0 (faces stay
// put). Out of line: called twice per hc row, the 32-bit sequence is large.
__attribute__((noinline)) void hc_face_transpose() {
    transpose_dest_init<true /*is_32bit*/, false /*transpose_of_faces*/>();
    transpose_dest<true /*is_32bit*/, false /*transpose_of_faces*/>(0);
}

// hc_row: the hc program in place over the LANE_TILES lane tiles (D0[, D1]).
ALWI void hc_row(const uint32_t* params, DataflowBuffer& lanes, DataflowBuffer& out) {
    {
        // zone "compute_hc_wait" dropped: kernel-text budget (profiler build)
        lanes.wait_front(mhc_post_hc::LANE_TILES);
        out.reserve_back(mhc_post_hc::LANE_TILES);
    }
    MaybeDeviceZoneScope("compute_hc");
    if constexpr (mhc_post_hc::NATURAL) {
        // stats tile (natural) -> face transpose -> regroup + hc program + restore (SFPU) -> face
        // transpose back -> the hc page (mhc_post_hc_sfpu.hpp NATURAL). Raw sequence, not a chain:
        // no kernel_lib element wraps transpose_dest around an SFPU body.
        reconfig_data_format_srca(cb_hc_lanes);
        pack_reconfig_data_format(cb_hc_out);
        tile_regs_acquire();
        copy_init(cb_hc_lanes);
        copy_tile(cb_hc_lanes, 0, 0);
        hc_face_transpose();
        mhc_post_hc::HcProgram<ckl::Dst::D0>::init();
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            ::ckernel::sfpu::calculate_mhc_hc_natural, 0, VectorMode::RC_custom, params)));
        hc_face_transpose();
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_hc_out);
        tile_regs_release();
    } else if constexpr (mhc_post_hc::LANE_TILES == 1) {
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            LoadLanes<ckl::Dst::D0>{lanes, 0},
            mhc_post_hc::HcProgram<ckl::Dst::D0>{params},
            PackLanes<ckl::Dst::D0>{out, 0});
    } else {
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            LoadLanes<ckl::Dst::D0>{lanes, 0},
            LoadLanes<ckl::Dst::D1>{lanes, 1},
            mhc_post_hc::HcProgram<ckl::Dst::D0>{params},
            PackLanes<ckl::Dst::D0>{out, 0},
            PackLanes<ckl::Dst::D1>{out, 1});
    }
    out.push_back(mhc_post_hc::LANE_TILES);
    lanes.pop_front(mhc_post_hc::LANE_TILES);
}

// mix_block: one chain over grid(n, extent). X block layout in the CB is [i][c] with the
// nominal block_c stride (ragged tail keeps nominal offsets).
ALWI void mix_block(
    uint32_t extent,
    DataflowBuffer& coef,
    DataflowBuffer& f,
    DataflowBuffer& x,
    DataflowBuffer& out,
    DataflowBuffer& xprime,
    DataflowBuffer& ss) {
    if constexpr (!fused) {
        ckl::eltwise_chain(
            ckl::IterationShape::grid(n, extent),
            CoefBcast{coef, COEF_POST_BASE},
            CopyF{f},
            MulPostF{},
            CoefBcast{coef, COEF_COMB_BASE + 0 * n},
            CopyX{x, 0 * block_c},
            MacCombX{ONE_F32_BITS},
            CoefBcast{coef, COEF_COMB_BASE + 1 * n},
            CopyX{x, 1 * block_c},
            MacCombX{ONE_F32_BITS},
            CoefBcast{coef, COEF_COMB_BASE + 2 * n},
            CopyX{x, 2 * block_c},
            MacCombX{ONE_F32_BITS},
            CoefBcast{coef, COEF_COMB_BASE + 3 * n},
            CopyX{x, 3 * block_c},
            MacCombX{ONE_F32_BITS},
            PackOut{out});
    } else {
        ckl::eltwise_chain(
            ckl::IterationShape::grid(n, extent),
            CoefBcast{coef, COEF_POST_BASE},
            CopyF{f},
            MulPostF{},
            CoefBcast{coef, COEF_COMB_BASE + 0 * n},
            CopyX{x, 0 * block_c},
            MacCombX{ONE_F32_BITS},
            CoefBcast{coef, COEF_COMB_BASE + 1 * n},
            CopyX{x, 1 * block_c},
            MacCombX{ONE_F32_BITS},
            CoefBcast{coef, COEF_COMB_BASE + 2 * n},
            CopyX{x, 2 * block_c},
            MacCombX{ONE_F32_BITS},
            CoefBcast{coef, COEF_COMB_BASE + 3 * n},
            CopyX{x, 3 * block_c},
            MacCombX{ONE_F32_BITS},
            PackOut{out},
            PackXprime{xprime},
            LoadSs{ss},
            MacSs{ONE_F32_BITS},
            PackSs{ss});
    }
}

// In1 (W slice) base offset of the current block inside the resident, block-major slice.
struct WBlockOffset {
    uint32_t base;
    ALWI uint32_t operator()(uint32_t /*block*/) const { return base; }
};

// project_block: proj_block = X'_block (1 x K) @ W_block (K x 1), K = n * extent.
ALWI void project_block(
    uint32_t extent, uint32_t w_base, DataflowBuffer& xprime, DataflowBuffer& w, DataflowBuffer& proj) {
    ckl::matmul_block<
        false,
        false,
        ckl::LastBlockTarget::Out,
        ckl::OutputCBLayout::SubblockMajor,
        ckl::matmul_config::InitMode::Short,
        ckl::InputPolicy::WaitAndPopPerKBlock,
        ckl::InputPolicy::NoWaitNoPop,
        ckl::NoPostCompute,
        ckl::NoPreKBlock,
        ckl::NoPostKBlock,
        0,
        ckl::NoKBlockInnerDimFn,
        ckl::NoIn0Source,
        WBlockOffset>(
        xprime,
        w,
        proj,
        proj,
        ckl::MatmulBlockShape::of(1, 1, 1, 1, n * extent, 1),
        {},
        {},
        1,
        0,
        {},
        {},
        {},
        WBlockOffset{w_base});
}

// Sum tiles 0..count-1 of a gather CB in slot order into `out_cb` (exact fp32 SFPU adds), the running
// sum resident in DEST for the whole fan-in: acc <- g0, then per slot tmp <- g[s], acc += tmp. Each
// slot is waited as the reader hands it over (per-slot arrival counters), so the adds overlap the
// arrivals instead of starting after the last one.
// Raw compute API, not an eltwise_chain (helper bypass, Perf 1): a chain acquires DEST per iteration
// and needs its element list at compile time, so a runtime-count sum had to ping-pong its running
// tile through an L1 CB (pack + unpack per slot, 3 chain calls). Same adds in the same order ->
// bit-identical stats; measured (perf_experiments/combine_tail "stream"): focus 107.5 -> 105.7 us,
// stats T160 C512 -7.5 %, collapse T64 C2048 -5.1 %.
// Perf 2 (perf_experiments/tail_hc "sum"): on Blackhole the running add is a load-macro stream
// (mhc_lm_add_d1: D0 <- 1.0 * D1 + D0, the same fma / rounding as add_binary_tile's SFPU add, same
// slot order -> bit-identical). sfpi / add_binary_tile issue load, load, add, store one per cycle; the
// macro overlaps the add and store with the next loads. Measured: the slowest leader's sum after its
// last slot 1,347 -> 942 ns (the straggler is often the leader's own slot 0, so every add of the row
// runs after it lands). Carve-out (inexpressible): other arches keep add_binary_tile (Blackhole
// load-macro / replay API).
template <uint32_t gather_cb, uint32_t out_cb>
ALWI void dest_sum(uint32_t count) {
    reconfig_data_format_srca(gather_cb);
    pack_reconfig_data_format(out_cb);
    tile_regs_acquire();
    copy_init(gather_cb);
#ifdef ARCH_BLACKHOLE
    MATH((::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::unused, DST_ACCUM_MODE>()));
#else
    add_binary_tile_init();
#endif
    for (uint32_t s = 0; s < count; ++s) {
        cb_wait_front(gather_cb, s + 1);
        copy_tile(gather_cb, s, s == 0 ? 0 : 1);
        if (s > 0) {
#ifdef ARCH_BLACKHOLE
            MHC_SFPU(::ckernel::sfpu::mhc_lm_add_d1);
#else
            add_binary_tile(0, 1, 0);
#endif
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(out_cb, 1);
    pack_tile(0, out_cb);
    cb_push_back(out_cb, 1);
    tile_regs_release();
}

}  // namespace

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t num_c = get_arg_val<uint32_t>(1);
    const uint32_t g_c = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks = (num_c + block_c - 1) / block_c;

    DataflowBuffer coef_buf(cb_coef_cols);
    DataflowBuffer f_buf(cb_f_tiles);
    DataflowBuffer x_buf(cb_x_tiles);
    DataflowBuffer out_buf(cb_out_tiles);

    if constexpr (!fused) {
        compute_kernel_hw_startup(cb_f_tiles, cb_x_tiles, cb_out_tiles);
        DataflowBuffer packed_buf(mix_fused ? cb_coef_packed : cb_coef_cols);
        for (uint32_t r = 0; r < num_rows; ++r) {
            {
                // zone "compute_coef_wait" dropped: kernel-text budget (profiler build)
                coef_buf.wait_front(n_coef_cols);  // load_coefficients (consumer side)
            }
            if constexpr (mix_fused) {
                // zone "compute_pack_coef" dropped: kernel-text budget (profiler build)
                packed_buf.reserve_back(n_packed);
                pack_coefficients(coef_buf, packed_buf);
                packed_buf.push_back(n_packed);
                packed_buf.wait_front(n_packed);
            }
            for (uint32_t b = 0; b < num_blocks; ++b) {
                const uint32_t extent = (num_c - b * block_c) < block_c ? (num_c - b * block_c) : block_c;
                {
                    MaybeDeviceZoneScope("compute_in_wait");
                    f_buf.wait_front(block_c);
                    x_buf.wait_front(n * block_c);
                }
                {
                    MaybeDeviceZoneScope("compute_mix");
                    if constexpr (mix_fused) {
                        mix_block_fused<false>(extent, out_buf, out_buf, out_buf);
                    } else {
                        mix_block(extent, coef_buf, f_buf, x_buf, out_buf, out_buf, out_buf);
                    }
                }
                f_buf.pop_front(block_c);
                x_buf.pop_front(n * block_c);
                // Ragged tail: keep the output window nominal (n * block_c pages per block) so the
                // writer's block window never wraps the CB ring. The padding pages are never written.
                if (extent < block_c) {
                    out_buf.reserve_back(n * (block_c - extent));
                    out_buf.push_back(n * (block_c - extent));
                }
            }
            if constexpr (mix_fused) {
                packed_buf.pop_front(n_packed);
            }
            coef_buf.pop_front(n_coef_cols);  // release_coefficients
        }
        return;
    } else {
        DataflowBuffer xprime_buf(cb_xprime_mm);
        DataflowBuffer w_buf(cb_w_slice);
        DataflowBuffer proj_block_buf(cb_proj_block);
        DataflowBuffer proj_run_buf(cb_proj_run);
        DataflowBuffer ss_run_buf(cb_ss_run);
        DataflowBuffer ss_row_buf(cb_ss_row);
        DataflowBuffer partial_buf(cb_partial);
        DataflowBuffer gather_buf(cb_gather);
        DataflowBuffer lead_out_buf(cb_lead_out);
        DataflowBuffer stats_row_buf(cb_stats_row);
        DataflowBuffer masks_buf(proj_passes > 1 ? cb_proj_masks : cb_xprime_mm);
        DataflowBuffer xmasked_buf(proj_passes > 1 ? cb_xprime_masked : cb_xprime_mm);

        constexpr auto geo0 = mhc_post_combine::Geometry::of(0, Q, COMBINE_FANIN);
        const auto geo = mhc_post_combine::Geometry::of(g_c, Q, COMBINE_FANIN);

        // Matmul boot (SrcOrder::Reverse); the chain elements re-init themselves.
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_xprime_mm, cb_w_slice, cb_proj_block);
        // The resident W slice arrives per block during the first row (the reader overlaps it with
        // that row's F / X reads); each first-row projection waits the cumulative count.
        if constexpr (proj_passes > 1) {
            masks_buf.wait_front(proj_passes);  // resident exact-projection masks
        }

        // collapse: the hc parameter block (RT args 3..) and the lane CBs.
        uint32_t hc_params[mhc_post_hc::NUM_PARAMS];
        if constexpr (collapse) {
            for (uint32_t i = 0; i < mhc_post_hc::NUM_PARAMS; ++i) {
                hc_params[i] = get_arg_val<uint32_t>(3 + i);
            }
        }
        DataflowBuffer y_buf(collapse ? cb_y_tiles : cb_xprime_mm);
        DataflowBuffer packed_buf(mix_fused ? cb_coef_packed : cb_coef_cols);
        DataflowBuffer hc_lanes_buf(collapse ? cb_hc_lanes : cb_xprime_mm);
        DataflowBuffer hc_out_buf(collapse ? cb_hc_out : cb_xprime_mm);
        // hc_row runs HC_LAG iterations after the row's stats are combined. With 1 (after the next
        // row's mix) this core's compute never waits on its writer's lane permute, which sits behind
        // the writer's cross-core credit / send waits; the writer polls for the result either way.
        uint32_t hc_due = 0, hc_produced = 0;  // stats rows awaiting hc: previous / this iteration
        auto stats_done = [&]() {
            if constexpr (collapse && HC_LAG == 0) {
                hc_row(hc_params, hc_lanes_buf, hc_out_buf);
            } else {
                ++hc_produced;
            }
        };
        auto run_due_hc = [&]() {
            if constexpr (collapse) {
                for (; hc_due > 0; --hc_due) {
                    hc_row(hc_params, hc_lanes_buf, hc_out_buf);
                }
                hc_due = hc_produced;
                hc_produced = 0;
            }
        };

        auto leader_combine = [&]() {
            {
                MaybeDeviceZoneScope("compute_lead_combine");  // includes the streamed slot waits
                if constexpr (geo0.groups > 1) {
                    dest_sum<cb_gather, cb_lead_out>(geo.size);
                } else {
                    dest_sum<cb_gather, cb_stats_row>(geo.size);
                }
            }
            gather_buf.wait_front(geo.slots);  // padding slots of a short group
            gather_buf.pop_front(geo.slots);
            if constexpr (geo0.groups == 1) {
                stats_done();
            }
        };
        auto root_combine = [&]() {
            if constexpr (geo0.groups > 1) {
                DataflowBuffer root_buf(cb_root_gather);
                {
                    MaybeDeviceZoneScope("compute_root_combine");  // includes the streamed slot waits
                    dest_sum<cb_root_gather, cb_stats_row>(geo0.groups);
                }
                root_buf.pop_front(geo0.groups);
                stats_done();
            }
        };

        // Row partial: running + block (exact SFPU add); the row's last contribution packs into
        // cb_partial for the writer.
        auto accumulate_partial = [&](bool first, bool last) {
            if (first) {
                if (last) {
                    ckl::eltwise_chain(
                        ckl::IterationShape::one_tile(),
                        CopyRun<cb_proj_block, ckl::Dst::D0>{proj_block_buf},
                        PackTo<cb_partial>{partial_buf});
                } else {
                    ckl::eltwise_chain(
                        ckl::IterationShape::one_tile(),
                        CopyRun<cb_proj_block, ckl::Dst::D0>{proj_block_buf},
                        PackTo<cb_proj_run>{proj_run_buf});
                }
            } else if (last) {
                ckl::eltwise_chain(
                    ckl::IterationShape::one_tile(),
                    CopyRun<cb_proj_run, ckl::Dst::D0>{proj_run_buf},
                    CopyRun<cb_proj_block, ckl::Dst::D1>{proj_block_buf},
                    AddD0D1{},
                    PackTo<cb_partial>{partial_buf});
            } else {
                ckl::eltwise_chain(
                    ckl::IterationShape::one_tile(),
                    CopyRun<cb_proj_run, ckl::Dst::D0>{proj_run_buf},
                    CopyRun<cb_proj_block, ckl::Dst::D1>{proj_block_buf},
                    AddD0D1{},
                    PackTo<cb_proj_run>{proj_run_buf});
            }
        };

        const uint32_t total = num_rows + 2 * COMBINE_LAG;
        for (uint32_t r = 0; r < total; ++r) {
            if (r < num_rows) {
                {
                    // zone "compute_coef_wait" dropped: kernel-text budget (profiler build)
                    coef_buf.wait_front(n_coef_cols);
                }
                {
                    // zone "compute_pack_coef" dropped: kernel-text budget (profiler build)
                    if constexpr (mix_fused) {
                        packed_buf.reserve_back(n_packed);
                        pack_coefficients(coef_buf, packed_buf);
                        packed_buf.push_back(n_packed);
                        packed_buf.wait_front(n_packed);
                    }
                    // ss_run <- 0 (row seed)
                    ckl::eltwise_chain(
                        ckl::IterationShape::one_tile(),
                        ckl::FillScalar<ckl::Dst::D0>{0.0f},
                        PackTo<cb_ss_run>{ss_run_buf});
                }
                for (uint32_t b = 0; b < num_blocks; ++b) {
                    const uint32_t extent = (num_c - b * block_c) < block_c ? (num_c - b * block_c) : block_c;
                    const uint32_t pad = block_c - extent;
                    {
                        MaybeDeviceZoneScope("compute_in_wait");
                        f_buf.wait_front(block_c);
                        x_buf.wait_front(n * block_c);
                    }
                    {
                        MaybeDeviceZoneScope("compute_mix");
                        if constexpr (mix_fused) {
                            mix_block_fused<true>(extent, out_buf, xprime_buf, ss_run_buf);
                        } else {
                            mix_block(extent, coef_buf, f_buf, x_buf, out_buf, xprime_buf, ss_run_buf);
                        }
                    }
                    f_buf.pop_front(block_c);
                    x_buf.pop_front(n * block_c);
                    if (pad) {
                        out_buf.reserve_back(n * pad);
                        out_buf.push_back(n * pad);
                        // keep cb_xprime_mm block-aligned (the matmul reads K contiguous tiles)
                        xprime_buf.reserve_back(n * pad);
                        xprime_buf.push_back(n * pad);
                    }
                    if constexpr (collapse) {
                        MaybeDeviceZoneScope("compute_collapse");
                        xprime_buf.wait_front(n * block_c);
                        if constexpr (mix_fused) {
                            collapse_block_fused(extent, packed_buf, xprime_buf, y_buf);
                        } else {
                            collapse_block(extent, coef_buf, xprime_buf, y_buf);
                        }
                        if (pad) {
                            y_buf.reserve_back(pad);
                            y_buf.push_back(pad);
                        }
                    }
                    const bool last_block = (b + 1 == num_blocks);
                    const uint32_t w_base = b * n * block_c;
                    if (r == 0) {
                        // zone "compute_w_wait" dropped: kernel-text budget (profiler build)
                        w_buf.wait_front(n * (w_base / n + extent));
                    }
                    MaybeDeviceZoneScope("compute_project");
                    if constexpr (proj_passes == 1) {
                        project_block(extent, w_base, xprime_buf, w_buf, proj_block_buf);
                        accumulate_partial(b == 0, last_block);
                        if (pad) {
                            xprime_buf.wait_front(n * pad);
                            xprime_buf.pop_front(n * pad);
                        }
                    } else {
                        // Exact projection: pass j multiplies the X' block by column mask j (columns
                        // j, j + 16; exact 0/1 FPU product) so every face of the matmul holds a
                        // single product, then accumulates in fp32 (exact SFPU add).
                        xprime_buf.wait_front(n * block_c);
                        for (uint32_t j = 0; j < proj_passes; ++j) {
                            ckl::eltwise_chain(
                                ckl::IterationShape::tiles(n * extent),
                                MaskX{xprime_buf, masks_buf, 0, j},
                                PackTo<cb_xprime_masked>{xmasked_buf});
                            if (pad) {
                                xmasked_buf.reserve_back(n * pad);
                                xmasked_buf.push_back(n * pad);
                            }
                            project_block(extent, w_base, xmasked_buf, w_buf, proj_block_buf);
                            if (pad) {
                                xmasked_buf.wait_front(n * pad);
                                xmasked_buf.pop_front(n * pad);
                            }
                            accumulate_partial(b == 0 && j == 0, last_block && j + 1 == proj_passes);
                        }
                        xprime_buf.pop_front(n * block_c);
                    }
                }
                // finish_row_partial: ss row sums -> column 0 of cb_ss_row
                // zone "compute_finish_row" dropped: kernel-text budget (profiler build)
                ckl::eltwise_chain(
                    ckl::IterationShape::one_tile(),
                    CopyRun<cb_ss_run, ckl::Dst::D0>{ss_run_buf},
                    SfpuRowSum<ckl::Dst::D0>{},
                    PackTo<cb_ss_row>{ss_row_buf});
                if constexpr (mix_fused) {
                    packed_buf.pop_front(n_packed);
                }
                coef_buf.pop_front(n_coef_cols);  // release_coefficients
            }
            // combine_row: leader combines row r - LAG, root combines row r - 2*LAG.
            if (geo.leader && r >= COMBINE_LAG && r - COMBINE_LAG < num_rows) {
                leader_combine();
            }
            if (r >= 2 * COMBINE_LAG && r - 2 * COMBINE_LAG < num_rows && geo.roots(r - 2 * COMBINE_LAG)) {
                root_combine();
            }
            run_due_hc();  // hc of the stats rows combined in the previous iteration
        }
        run_due_hc();  // drain: the last iteration's stats rows
    }
}
