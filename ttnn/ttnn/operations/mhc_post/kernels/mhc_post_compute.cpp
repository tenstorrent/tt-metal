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
//   tile ping-pongs through cb_comb_acc) -> cb_lead_out (two-level) / cb_stats_row.
//
// Helper gaps (recorded, not bypassed): no kernel_lib element wraps the SFPU row reduce, so
// SfpuRowSum is a 5-line UnaryOp element (init/exec only — the chain still owns DEST sync, CB
// lifecycle and reconfig). The FPU reduce helper is rejected: it reads fp32 through srcA/B (tf32),
// ~1e-3 rel error vs the 1e-6 SUMSQ gate.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/ternary/ternary.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/ttnn/operations/mhc_post/kernels/mhc_post_combine.hpp"

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
constexpr uint32_t cb_comb_acc = get_compile_time_arg_val(FB + 11);
constexpr uint32_t cb_stats_row = get_compile_time_arg_val(FB + 12);
constexpr uint32_t Q = get_compile_time_arg_val(FB + 13);
constexpr uint32_t COMBINE_FANIN = get_compile_time_arg_val(FB + 14);
constexpr uint32_t COMBINE_LAG = get_compile_time_arg_val(FB + 16);
constexpr uint32_t proj_passes = get_compile_time_arg_val(FB + 22);
constexpr uint32_t cb_proj_masks = get_compile_time_arg_val(FB + 23);
constexpr uint32_t cb_xprime_masked = get_compile_time_arg_val(FB + 24);

static_assert(n == 4, "mhc_post: the mix chain is written for n = 4 streams");

constexpr uint32_t ONE_F32_BITS = 0x3f800000u;  // Addcmul value = 1.0 (exact)
constexpr uint32_t COEF_POST_BASE = 0;
constexpr uint32_t COEF_COMB_BASE = n;

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

// ---------------------------------------------------------------------------
// Block operations
// ---------------------------------------------------------------------------

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

// Sum tiles 0..count-1 of a staged gather CB in index order into `out_cb` (exact SFPU adds):
// acc <- g0; one streaming chain acc <- acc + g[s] for s = 1..count-1 (acc ping-pongs through
// cb_comb_acc, fp32 / UnpackToDestFp32, so every partial stays exact); out <- acc. Three chain
// calls regardless of the fan-in.
template <uint32_t gather_cb>
using GatherAt = ckl::CopyTile<
    ckl::input(
        gather_cb,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Block,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset),
    ckl::Dst::D1>;

template <uint32_t gather_cb, uint32_t out_cb>
ALWI void sum_slots(uint32_t count, DataflowBuffer& gather, DataflowBuffer& acc, DataflowBuffer& out) {
    if (count == 1) {
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(), CopyAt<gather_cb, ckl::Dst::D0>{gather, 0}, PackTo<out_cb>{out});
        return;
    }
    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(), CopyAt<gather_cb, ckl::Dst::D0>{gather, 0}, PackTo<cb_comb_acc>{acc});
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(count - 1),
        CopyRun<cb_comb_acc, ckl::Dst::D0>{acc},
        GatherAt<gather_cb>{gather, 1},
        AddD0D1{},
        PackTo<cb_comb_acc>{acc});
    ckl::eltwise_chain(ckl::IterationShape::one_tile(), CopyRun<cb_comb_acc, ckl::Dst::D0>{acc}, PackTo<out_cb>{out});
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
        for (uint32_t r = 0; r < num_rows; ++r) {
            coef_buf.wait_front(n_coef_cols);  // load_coefficients (consumer side)
            for (uint32_t b = 0; b < num_blocks; ++b) {
                const uint32_t extent = (num_c - b * block_c) < block_c ? (num_c - b * block_c) : block_c;
                f_buf.wait_front(block_c);
                x_buf.wait_front(n * block_c);
                mix_block(extent, coef_buf, f_buf, x_buf, out_buf, out_buf, out_buf);
                f_buf.pop_front(block_c);
                x_buf.pop_front(n * block_c);
                // Ragged tail: keep the output window nominal (n * block_c pages per block) so the
                // writer's block window never wraps the CB ring. The padding pages are never written.
                if (extent < block_c) {
                    out_buf.reserve_back(n * (block_c - extent));
                    out_buf.push_back(n * (block_c - extent));
                }
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
        DataflowBuffer comb_acc_buf(cb_comb_acc);
        DataflowBuffer stats_row_buf(cb_stats_row);
        DataflowBuffer masks_buf(proj_passes > 1 ? cb_proj_masks : cb_xprime_mm);
        DataflowBuffer xmasked_buf(proj_passes > 1 ? cb_xprime_masked : cb_xprime_mm);

        constexpr auto geo0 = mhc_post_combine::Geometry::of(0, Q, COMBINE_FANIN);
        const auto geo = mhc_post_combine::Geometry::of(g_c, Q, COMBINE_FANIN);

        // Matmul boot (SrcOrder::Reverse); the chain elements re-init themselves.
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_xprime_mm, cb_w_slice, cb_proj_block);
        w_buf.wait_front(n * num_c);  // resident W slice for the whole kernel
        if constexpr (proj_passes > 1) {
            masks_buf.wait_front(proj_passes);  // resident exact-projection masks
        }

        auto leader_combine = [&]() {
            gather_buf.wait_front(geo.slots);
            if constexpr (geo0.groups > 1) {
                sum_slots<cb_gather, cb_lead_out>(geo.size, gather_buf, comb_acc_buf, lead_out_buf);
            } else {
                sum_slots<cb_gather, cb_stats_row>(geo.size, gather_buf, comb_acc_buf, stats_row_buf);
            }
            gather_buf.pop_front(geo.slots);
        };
        auto root_combine = [&]() {
            if constexpr (geo0.groups > 1) {
                DataflowBuffer root_buf(cb_root_gather);
                root_buf.wait_front(geo0.groups);
                sum_slots<cb_root_gather, cb_stats_row>(geo0.groups, root_buf, comb_acc_buf, stats_row_buf);
                root_buf.pop_front(geo0.groups);
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
                coef_buf.wait_front(n_coef_cols);
                // ss_run <- 0 (row seed)
                ckl::eltwise_chain(
                    ckl::IterationShape::one_tile(),
                    ckl::FillScalar<ckl::Dst::D0>{0.0f},
                    PackTo<cb_ss_run>{ss_run_buf});
                for (uint32_t b = 0; b < num_blocks; ++b) {
                    const uint32_t extent = (num_c - b * block_c) < block_c ? (num_c - b * block_c) : block_c;
                    const uint32_t pad = block_c - extent;
                    f_buf.wait_front(block_c);
                    x_buf.wait_front(n * block_c);
                    mix_block(extent, coef_buf, f_buf, x_buf, out_buf, xprime_buf, ss_run_buf);
                    f_buf.pop_front(block_c);
                    x_buf.pop_front(n * block_c);
                    if (pad) {
                        out_buf.reserve_back(n * pad);
                        out_buf.push_back(n * pad);
                        // keep cb_xprime_mm block-aligned (the matmul reads K contiguous tiles)
                        xprime_buf.reserve_back(n * pad);
                        xprime_buf.push_back(n * pad);
                    }
                    const bool last_block = (b + 1 == num_blocks);
                    const uint32_t w_base = b * n * block_c;
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
                ckl::eltwise_chain(
                    ckl::IterationShape::one_tile(),
                    CopyRun<cb_ss_run, ckl::Dst::D0>{ss_run_buf},
                    SfpuRowSum<ckl::Dst::D0>{},
                    PackTo<cb_ss_row>{ss_row_buf});
                coef_buf.pop_front(n_coef_cols);  // release_coefficients
            }
            // combine_row: leader combines row r - LAG, root combines row r - 2*LAG.
            if (geo.leader && r >= COMBINE_LAG && r - COMBINE_LAG < num_rows) {
                leader_combine();
            }
            if (r >= 2 * COMBINE_LAG && r - 2 * COMBINE_LAG < num_rows && geo.roots(r - 2 * COMBINE_LAG)) {
                root_combine();
            }
        }
    }
}
