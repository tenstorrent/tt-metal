// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Cross-core W-split compute kernel for rms_norm (op_design.md §1 lamp 2, §5).
//
// The reduced dim W is split across a group of K cores (WIDTH/BLOCK shard, or a
// logical W-split of a wide interleaved input). Each core holds a W-slice of the
// same tile-rows resident in L1 (zero-copy sharded cb_x_in). The RMS statistic
// spans all of W, so it is a CROSS-CORE reduction:
//
//   pass 1 (local) : per tile-row, Σ_slice x²·(1/W)             -> cb_stat_local (partial)
//   cross-core     : writer gathers the K partials to the group MASTER
//   master combine : Σ_k partial_k  -> mean ; (+eps, rsqrt)     -> cb_stat_handoff (1/RMS)
//   cross-core     : writer broadcasts 1/RMS back to the group  -> cb_stat_global
//   pass 2 (local) : per tile-row, x·rstd·gamma                 -> cb_out (sharded)
//
// One gather+broadcast round per tile-row (fully synchronous, monotone counter
// semaphores in the writer) so cb_gather stays K tiles — no CB grows with the
// tile-row count. The compute<->writer handoff is via cb_stat_local (compute ->
// writer), cb_gather (writer -> master compute), cb_stat_handoff (master compute
// -> writer), cb_stat_global (writer -> compute). cb_stat_handoff is a SEPARATE
// CB from cb_stat_global (never reuse one CB for both, §7 two-consumer trap).
//
// Reused helpers: square / reduce (accumulate) / eltwise_chain mul (indexed
// resident access, same lower-level form as the R3 resident path). The one
// raw-LLK block is the master's K-partial fold + (+eps, rsqrt): no kernel-lib
// helper reduces across N distinct CB tiles element-wise with an SFPU postop —
// this is the sanctioned tensix_all_reduce fold pattern
// (examples/tensix_all_reduce _REDUCE_KERNEL), with the RMS finalize appended
// in the same DST window (identical to the streaming path's transform_in_place
// lambda body: add_unary_tile(eps) then rsqrt_tile).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t cb_x_sticks = 0;      // RM input: tile-padded sticks (reader loopback) -> tilize
constexpr uint32_t cb_x_in = 1;          // sharded input W-slice (zero-copy, resident) OR tilize output (RM)
constexpr uint32_t cb_scaler = 2;        // 1/W reduce scaler (+partial), bf16
constexpr uint32_t cb_gamma = 3;         // gamma W-slice tiles (held)
constexpr uint32_t cb_gamma_sticks = 4;  // RM-gamma stick input (tilized -> cb_gamma)
constexpr uint32_t cb_gather = 5;        // master: K partials gathered (fp32); R6c two-stage: NX/row
constexpr uint32_t cb_stat_handoff = 6;  // master: 1/RMS -> writer broadcast (fp32)
constexpr uint32_t cb_stat_global = 7;   // 1/RMS received (fp32)
constexpr uint32_t cb_rowpartial = 10;   // R6c two-stage: row-leader stage-1 fold output (-> writer)
constexpr uint32_t cb_gather2 = 11;      // R6c two-stage: root stage-2 fan-in (NY row-partials)
constexpr uint32_t cb_out = 16;          // sharded output W-slice (zero-copy) OR untilize input (RM)
constexpr uint32_t cb_out_sticks = 17;   // RM output: tile-padded sticks (untilize) -> writer loopback
constexpr uint32_t cb_xsq = 24;          // x^2 (pass-1 intermediate)
constexpr uint32_t cb_stat_local = 25;   // per-tile-row local partial Σx²·(1/W) (fp32)
constexpr uint32_t cb_norm = 26;         // x·rstd (pass-2 intermediate)
}  // namespace

void kernel_main() {
    constexpr uint32_t PER_W_T = get_compile_time_arg_val(0);   // W-tiles this core holds
    constexpr uint32_t HT_LOCAL = get_compile_time_arg_val(1);  // tile-rows this core holds
    constexpr uint32_t K = get_compile_time_arg_val(2);         // group size (cores over W)
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(3) != 0;
    constexpr bool HAS_PARTIAL_W = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t eps_bits = get_compile_time_arg_val(5);
    // RM gamma -> compute tilizes cb_gamma_sticks (vwt one-tile-wide blocks) into
    // cb_gamma before holding it resident. TILE gamma -> reader already filled
    // cb_gamma (tilize skipped). Cross-core mirror of the interleaved knob-turn.
    constexpr bool GAMMA_IS_RM = get_compile_time_arg_val(6) != 0;
    // X_ZERO_COPY=1: cb_x_in is backed zero-copy on the resident sharded W-slice, so
    // compute self-arms it (no external producer). X_ZERO_COPY=0 (logical
    // wide-interleaved / decode W-split): the reader reads this core's W/K slice from
    // interleaved DRAM into cb_x_in, so compute only waits on the reader's push.
    constexpr bool X_ZERO_COPY = get_compile_time_arg_val(7) != 0;
    // IS_RM (Refinement 4b): the input is a ROW-MAJOR sharded W-slice whose width is an
    // arbitrary multiple of the RM granule (8/4 el), NOT a whole number of 32-wide
    // tiles. The reader loopback-repacks the resident RM shard sticks into tile-padded
    // cb_x_sticks; compute tilizes them into cb_x_in (PER_W_T = ceil(sw/32) padded
    // tiles per tile-row), runs the SAME cross-core combine with a PER-CORE partial
    // scaler (every core's last W-tile is sub-tile-wide), then untilizes cb_out back
    // to tile-padded cb_out_sticks for the writer to loopback into the resident RM
    // output shard. PER_W_T here is per_w_t_padded; vwt is the reduce tile count
    // (ceil(valid_cols/32)); is_partial_holder is per-core (valid_cols % 32 != 0).
    constexpr bool IS_RM = get_compile_time_arg_val(8) != 0;
    // C_ROWS (Refinement 6a lever 1): batch C tile-rows' stats per cross-core round so the
    // fully-synchronous round latency (~3150 ns, dominant for BLOCK's HT_LOCAL=32) amortizes
    // over C rows — sync rounds drop from HT_LOCAL to ceil(HT_LOCAL/C). Compute produces C
    // local partials (cb_stat_local depth 2*C), the writer gathers K*C, the master folds C
    // rstds, broadcasts C, then pass-2 covers the C tile-rows. C=1 is byte-identical to the
    // per-tile-row scheme (R4); the host sets C>1 only on the pure tiled resident-shard path
    // (RM / logical-out-to-DRAM keep C=1, their per-tile-row drain unchanged). This tiled
    // path is the only one batched; the RM path above always runs at C=1.
    constexpr uint32_t C_ROWS = get_compile_time_arg_val(9);
    // PIPELINE_LOOKAHEAD (Refinement 6b lever 2): when set, the compute issues batch r+1's
    // pass 1 BEFORE batch r's fold/pass 2, so the local reduce overlaps the writer's
    // synchronous cross-core round. It is the COMPLEMENTARY STEP to the gather compaction
    // (lever 1): with the full-tile gather the round dwarfs the compute so overlap is flat
    // (ablation mode0: pipe-on 119036 == pipe-off 119027), but once the compaction shrinks the
    // gather (mode 2) the round is small enough that overlapping compute WINS — BLOCK 8x8
    // 113810 (pipe-off) -> 107995 (pipe-on), 1.05x on top of compaction. Ships ON (idx 10=1) on
    // the multi-round tiled path; degenerates byte-identically to the R6a sequential order for
    // single-round groups (WIDTH, num_rounds==1) and when the host sets it 0. cb_stat_local is
    // 2*C deep (two rounds in flight) and the writer/semaphore protocol + fixed-base
    // cb_gather/cb_stat_global addressing are UNCHANGED — only the compute issue order changes.
    constexpr bool PIPELINE_LOOKAHEAD = get_compile_time_arg_val(10) != 0;
    // TWO_STAGE (Refinement 6c): hierarchical gather over a 2D group (NX x NY). Row-leaders (incl
    // the root) fold their row's NX partials -> a row-partial (stage 1); the root folds the NY
    // row-partials + finalize (+eps, rsqrt) -> 1/RMS (stage 2). Host-gated single-round (C=1). The
    // per-tile-row math (pass 1 / pass 2) is byte-identical to the flat path; only the fold
    // topology differs (distributed across row-leaders instead of one master).
    constexpr bool TWO_STAGE = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t NX = get_compile_time_arg_val(12);  // stage-1 fold count (cores per grid row)
    constexpr uint32_t NY = get_compile_time_arg_val(13);  // stage-2 fold count (grid rows)
    // TWO_PHASE_FOLD (Refinement 6e): distribute the master's serial K-partial fold across up to
    // NUM_FOLDERS = min(C_ROWS, K) "folder" cores by tile-index. Each folder gathers+folds a
    // disjoint set of the C batched tile-rows -> its finalized 1/RMS, scatters them to the root,
    // and the root assembles + mcasts back. Only the FOLD moves off the master (pass 1 / pass 2 /
    // the round-batch structure are unchanged); this attacks the master serial fold arithmetic
    // (in-op ablation: 28% of BLOCK 8x8, fully on the critical path — the R6d two_phase winner).
    // Host-gated to the pure tiled BLOCK path with C>1 multi-round (Ht_local % C == 0); C=1 /
    // WIDTH / RM / logical / two-stage keep the byte-identical flat master fold below.
    constexpr bool TWO_PHASE_FOLD = get_compile_time_arg_val(14) != 0;
    constexpr uint32_t NUM_FOLDERS = get_compile_time_arg_val(15);

    const uint32_t vwt = get_arg_val<uint32_t>(0);  // valid W-tiles (<= PER_W_T)
    const uint32_t is_partial_holder = get_arg_val<uint32_t>(1);
    const uint32_t is_master = get_arg_val<uint32_t>(2);
    const uint32_t is_row_leader = get_arg_val<uint32_t>(3);  // R6c: role 1 (x0 column, y != y0)
    const uint32_t is_folder = get_arg_val<uint32_t>(4);      // R6e: this core folds (slice_index < NUM_FOLDERS)
    const uint32_t owned_max = get_arg_val<uint32_t>(5);      // R6e: max owned rows this folder handles per round

    compute_kernel_hw_startup(cb_x_in, cb_scaler, cb_out);

    constexpr auto one_tile = ckl::EltwiseShape::of(1, 1);
    constexpr auto partial_scaler_sel = ckl::ReducePartialScaler::last_tile_at(1);

    // ============================ RM-input sharded path (Refinement 4b) ============================
    // Same cross-core scheme as the tiled path below; only the boundary changes — the
    // resident W-slice arrives as RM sticks (tile-padded in cb_x_sticks by the reader),
    // so compute tilizes each tile-row before pass 1 and untilizes cb_out after pass 2.
    if constexpr (IS_RM) {
        if constexpr (HAS_GAMMA) {
            // gamma W-slice arrives RM (RM input never carries TILE gamma — INVALID);
            // the reader pushed `vwt` one-tile-wide stick blocks (the valid tiles, tile
            // wt covering global columns [(g0+wt)*32, ...)). Tilize once, hold.
            ckl::tilize<1, cb_gamma_sticks, cb_gamma>(vwt);
            cb_wait_front(cb_gamma, vwt);
        }

        for (uint32_t t = 0; t < HT_LOCAL; ++t) {
            // Tilize this tile-row's PER_W_T padded tiles (reader loopback-filled
            // cb_x_sticks) into cb_x_in; held across both passes of this tile-row.
            ckl::tilize<PER_W_T, cb_x_sticks, cb_x_in>(1);
            cb_wait_front(cb_x_in, PER_W_T);

            // ---------- Pass 1: local partial Σ_slice x²·(1/W) over vwt reduce-tiles ----------
            for (uint32_t w = 0; w < vwt; ++w) {
                ckl::eltwise_chain(
                    one_tile,
                    ckl::BinaryFpu<
                        cb_x_in,
                        cb_x_in,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::None,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block,
                        ckl::OperandKind::Block,
                        ckl::TileOffset::Set,
                        ckl::TileOffset::Set>{w, w},
                    ckl::PackTile<cb_xsq, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
            }
            const ckl::ReducePartialScaler ps =
                is_partial_holder ? partial_scaler_sel : ckl::ReducePartialScaler::none();
            ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, cb_xsq, cb_scaler, cb_stat_local>(
                ckl::ReduceInputBlockShape::of(1, vwt, 1),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::Accumulate::at(cb_stat_local, 0),
                ckl::NoOp{},
                ps);

            // ---------- Master: fold K partials -> mean; (+eps, rsqrt) -> 1/RMS ----------
            if (is_master) {
                cb_wait_front(cb_gather, K);
                cb_reserve_back(cb_stat_handoff, 1);
                reconfig_data_format(cb_gather, cb_gather);
                pack_reconfig_data_format(cb_stat_handoff);
                tile_regs_acquire();
                uint32_t first_pair = 0;
                if (K & 1u) {
                    copy_tile_to_dst_init_short(cb_gather);
                    copy_tile(cb_gather, 0, 0);
                    first_pair = 1;
                }
                if (K > 1) {
                    add_tiles_init(cb_gather, cb_gather, /*acc_to_dest=*/true);
                    for (uint32_t k = first_pair; k < K; k += 2) {
                        add_tiles(cb_gather, cb_gather, k, k + 1, 0);
                    }
                }
                binop_with_scalar_tile_init();
                add_unary_tile(0, eps_bits);
                rsqrt_tile_init();
                rsqrt_tile(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_stat_handoff);
                tile_regs_release();
                cb_push_back(cb_stat_handoff, 1);
                cb_pop_front(cb_gather, K);
            }

            // ---------- Pass 2: x·rstd·gamma over the whole PER_W_T-tile W-slice ----------
            cb_wait_front(cb_stat_global, 1);
            for (uint32_t w = 0; w < PER_W_T; ++w) {
                ckl::eltwise_chain(
                    one_tile,
                    ckl::BinaryFpu<
                        cb_x_in,
                        cb_stat_global,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Col,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::InputLifecycle::HeldBulk,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block,
                        ckl::OperandKind::Col,
                        ckl::TileOffset::Set,
                        ckl::TileOffset::Unset>{w, 0},
                    ckl::PackTile<cb_norm, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});

                if (HAS_GAMMA && (w < vwt)) {
                    // gamma applies to the vwt valid tiles (which hold the output columns
                    // [phase, phase+valid_cols)); tiles w >= vwt are all-pad output
                    // (discarded on writeback), so a plain copy suffices there.
                    ckl::eltwise_chain(
                        one_tile,
                        ckl::BinaryFpu<
                            cb_norm,
                            cb_gamma,
                            ckl::BinaryFpuOp::Mul,
                            ckl::BroadcastDim::Row,
                            ckl::InputLifecycle::Streaming,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::BinaryDataFormatReconfig::Input,
                            ckl::Dst::D0,
                            ckl::OperandKind::Scalar,
                            ckl::OperandKind::Row,
                            ckl::TileOffset::Unset,
                            ckl::TileOffset::Set>{0, w},
                        ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
                } else {
                    ckl::copy<cb_norm, cb_out>(one_tile);
                }
            }
            cb_pop_front(cb_stat_global, 1);

            // Untilize the tile-row's PER_W_T output tiles -> cb_out_sticks (tile-padded
            // sticks); the writer loopback-copies the valid columns into the RM shard.
            ckl::untilize<PER_W_T, cb_out, cb_out_sticks>(1);

            cb_pop_front(cb_x_in, PER_W_T);
        }

        if constexpr (HAS_GAMMA) {
            cb_pop_front(cb_gamma, vwt);
        }
        cb_pop_front(cb_scaler, 2);  // RM always prepares full(tile0)+partial-or-full(tile1)
        return;
    }
    // ============================ end RM-input sharded path ============================

    // Arm the input W-slice once (whole slice held; indexed access across both passes).
    const uint32_t shard_tiles = HT_LOCAL * PER_W_T;
    if constexpr (X_ZERO_COPY) {
        cb_reserve_back(cb_x_in, shard_tiles);  // self-arm the resident zero-copy shard
        cb_push_back(cb_x_in, shard_tiles);
    }
    cb_wait_front(cb_x_in, shard_tiles);  // (reader is the producer on the logical path)

    if constexpr (HAS_GAMMA) {
        if constexpr (GAMMA_IS_RM) {
            // Tilize the vwt one-tile-wide stick blocks the reader pushed into
            // cb_gamma (BLOCK_SIZE=1 so num_blocks=vwt is a runtime count). This is
            // the sole producer of cb_gamma on the RM-gamma path.
            ckl::tilize<1, cb_gamma_sticks, cb_gamma>(vwt);
        }
        cb_wait_front(cb_gamma, vwt);  // gamma W-slice held (read/tilized once)
    }

    // Round-batching (Refinement 6a lever 1): one cross-core round exchanges C_ROWS
    // tile-rows' partials. Rounds = ceil(HT_LOCAL / C_ROWS); the last is short when
    // C_ROWS does not divide HT_LOCAL. C_ROWS=1 reduces exactly to the R4 per-row loop.
    const uint32_t num_rounds = (HT_LOCAL + C_ROWS - 1) / C_ROWS;
    auto c_of = [&](uint32_t r) -> uint32_t {
        const uint32_t c = HT_LOCAL - r * C_ROWS;
        return (c > C_ROWS) ? C_ROWS : c;
    };

    // ---------- Pass 1: local partial Σ_slice x²·(1/W) for a batch of C_this tile-rows ----------
    // Square the vwt valid W-tiles into cb_xsq, then reduce the whole block in ONE call
    // (cols=vwt) per tile-row — each reduce pushes one partial into cb_stat_local (depth 2*C).
    // The partial-holder routes the partial scaler to the block's last tile. A lambda so the
    // round/compute pipeline (R6b lever 2) can issue the NEXT round's pass 1 before the current
    // round's fold/pass 2 — overlapping this compute with the writer's synchronous transport.
    auto do_pass1 = [&](uint32_t base_t, uint32_t C_this) {
        for (uint32_t cc = 0; cc < C_this; ++cc) {
            const uint32_t t = base_t + cc;
            for (uint32_t w = 0; w < vwt; ++w) {
                const uint32_t xin_off = t * PER_W_T + w;
                ckl::eltwise_chain(
                    one_tile,
                    ckl::BinaryFpu<
                        cb_x_in,
                        cb_x_in,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::None,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block,
                        ckl::OperandKind::Block,
                        ckl::TileOffset::Set,
                        ckl::TileOffset::Set>{xin_off, xin_off},
                    ckl::PackTile<cb_xsq, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
            }
            const ckl::ReducePartialScaler ps =
                (HAS_PARTIAL_W && is_partial_holder) ? partial_scaler_sel : ckl::ReducePartialScaler::none();
            ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, cb_xsq, cb_scaler, cb_stat_local>(
                ckl::ReduceInputBlockShape::of(1, vwt, 1),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::Accumulate::at(cb_stat_local, 0),
                ckl::NoOp{},
                ps);
        }
        // reduce pushed C_this partials into cb_stat_local. Writer gathers them to the master.
    };

    // ---------- Master: fold K partials -> mean; (+eps, rsqrt) -> 1/RMS, per row ----------
    auto do_fold = [&](uint32_t C_this) {
        cb_wait_front(cb_gather, K * C_this);  // writer gathered K partials × C_this rows
        cb_reserve_back(cb_stat_handoff, C_this);
        // Raw-LLK fold needs the data-format reconfig the helpers do implicitly: pass 1 left
        // the unpacker configured for cb_xsq/cb_scaler; the fold reads cb_gather (fp32) and
        // packs cb_stat_handoff (fp32). Without this the unpacker-A src-format LLK_ASSERT
        // trips (hang). One reconfig covers all C_this folds (same src/dst formats).
        reconfig_data_format(cb_gather, cb_gather);
        pack_reconfig_data_format(cb_stat_handoff);
        for (uint32_t cc = 0; cc < C_this; ++cc) {
            // Row cc's K partials are the contiguous cb_gather tiles [cc*K, cc*K + K).
            const uint32_t g = cc * K;
            tile_regs_acquire();
            // dst0 = Σ_{k=0..K-1} cb_gather[g + k]  (col0 holds the per-row partial;
            // other columns are ignored downstream via BroadcastDim::Col).
            uint32_t first_pair = 0;
            if (K & 1u) {
                copy_tile_to_dst_init_short(cb_gather);
                copy_tile(cb_gather, g + 0, 0);
                first_pair = 1;
            }
            if (K > 1) {
                add_tiles_init(cb_gather, cb_gather, /*acc_to_dest=*/true);
                for (uint32_t k = first_pair; k < K; k += 2) {
                    add_tiles(cb_gather, cb_gather, g + k, g + k + 1, 0);
                }
            }
            // RMS finalize on the summed mean (same body as the streaming
            // transform_in_place lambda): rstd = rsqrt(mean + eps).
            binop_with_scalar_tile_init();
            add_unary_tile(0, eps_bits);
            rsqrt_tile_init();
            rsqrt_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_stat_handoff);  // sequential pack -> handoff slot cc
            tile_regs_release();
        }
        cb_push_back(cb_stat_handoff, C_this);
        cb_pop_front(cb_gather, K * C_this);
    };

    // ---------- Pass 2: x·rstd·gamma over the W-slice for a batch of C_this tile-rows ----------
    auto do_pass2 = [&](uint32_t base_t, uint32_t C_this) {
        cb_wait_front(cb_stat_global, C_this);  // C_this 1/RMS tiles (broadcast landed)
        for (uint32_t cc = 0; cc < C_this; ++cc) {
            const uint32_t t = base_t + cc;
            for (uint32_t w = 0; w < PER_W_T; ++w) {
                const uint32_t xin_off = t * PER_W_T + w;
                // x·rstd (rstd is REDUCE_ROW -> column-shaped -> BroadcastDim::Col). rstd for
                // row cc lives at cb_stat_global tile cc (broadcast wrote C_this contiguous
                // tiles); read it via the Col operand's TileOffset::Set base. CallerManaged:
                // this loop owns the cb_stat_global wait/pop (C_this) around the whole batch.
                ckl::eltwise_chain(
                    one_tile,
                    ckl::BinaryFpu<
                        cb_x_in,
                        cb_stat_global,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Col,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block,
                        ckl::OperandKind::Col,
                        ckl::TileOffset::Set,
                        ckl::TileOffset::Set>{xin_off, cc},
                    ckl::PackTile<cb_norm, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});

                if (HAS_GAMMA && (w < vwt)) {
                    // norm·gamma (gamma is [1,W] -> row-shaped -> BroadcastDim::Row).
                    ckl::eltwise_chain(
                        one_tile,
                        ckl::BinaryFpu<
                            cb_norm,
                            cb_gamma,
                            ckl::BinaryFpuOp::Mul,
                            ckl::BroadcastDim::Row,
                            ckl::InputLifecycle::Streaming,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::BinaryDataFormatReconfig::Input,
                            ckl::Dst::D0,
                            ckl::OperandKind::Scalar,
                            ckl::OperandKind::Row,
                            ckl::TileOffset::Unset,
                            ckl::TileOffset::Set>{0, w},
                        ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
                } else {
                    // no gamma, or a trailing padding tile (output discarded on read-back).
                    ckl::copy<cb_norm, cb_out>(one_tile);
                }
            }
        }
        cb_pop_front(cb_stat_global, C_this);
    };

    // R6c two-stage fold: sum `count` fp32 tiles from `in_cb` into DST[0] (associative partial
    // sum; col-0 carries the reduce result, other columns ignored downstream via BroadcastDim::Col)
    // and pack to `out_cb`. `finalize` appends the RMS finish (+eps, rsqrt) — same DST body as the
    // flat do_fold / streaming transform_in_place. Stage 1 (row-leader): fold NX, no finalize ->
    // cb_rowpartial. Stage 2 (root): fold NY, finalize -> cb_stat_handoff. C=1 (single round).
    auto fold_tiles = [&](uint32_t in_cb, uint32_t out_cb, uint32_t count, bool finalize) {
        cb_wait_front(in_cb, count);
        cb_reserve_back(out_cb, 1);
        reconfig_data_format(in_cb, in_cb);
        pack_reconfig_data_format(out_cb);
        tile_regs_acquire();
        uint32_t first_pair = 0;
        if (count & 1u) {
            copy_tile_to_dst_init_short(in_cb);
            copy_tile(in_cb, 0, 0);
            first_pair = 1;
        }
        if (count > 1) {
            add_tiles_init(in_cb, in_cb, /*acc_to_dest=*/true);
            for (uint32_t k = first_pair; k < count; k += 2) {
                add_tiles(in_cb, in_cb, k, k + 1, 0);
            }
        }
        if (finalize) {
            binop_with_scalar_tile_init();
            add_unary_tile(0, eps_bits);
            rsqrt_tile_init();
            rsqrt_tile(0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        cb_push_back(out_cb, 1);
        cb_pop_front(in_cb, count);
    };

    // R6c two-stage (hierarchical) gather — single round (host-gated C=1, HT_LOCAL=1):
    //   pass 1 -> local partial; row-leaders (incl root) fold their row (stage 1); root folds the
    //   row-leaders (stage 2, +finalize); pass 2. The writer routes the cross-core traffic; the
    //   per-tile-row math is identical to the flat path. Non-two-stage falls through to the round
    //   loop below (byte-identical).
    if constexpr (TWO_STAGE) {
        do_pass1(0, 1);
        if (is_row_leader || is_master) {
            fold_tiles(cb_gather, cb_rowpartial, NX, /*finalize=*/false);  // stage 1
        }
        if (is_master) {
            fold_tiles(cb_gather2, cb_stat_handoff, NY, /*finalize=*/true);  // stage 2 (+eps, rsqrt)
        }
        do_pass2(0, 1);
        cb_pop_front(cb_x_in, shard_tiles);
        if constexpr (HAS_GAMMA) {
            cb_pop_front(cb_gamma, vwt);
        }
        cb_pop_front(cb_scaler, HAS_PARTIAL_W ? 2 : 1);
        return;
    }

    // ---------- Refinement 6e: two-phase (tile-index) distributed fold ----------
    // Every folder (slice_index < NUM_FOLDERS) folds its `owned_max` disjoint tile-rows per
    // round (K partials each -> +eps, rsqrt) into cb_rowpartial (reused as the folder-local
    // finalized-rstd buffer; two-stage is off on this path). The writer scatters those rstds to
    // the root, which assembles all C and mcasts back. Non-folders skip the fold. do_pass1 /
    // do_pass2 are byte-identical to the flat path — only the fold is distributed off the master.
    if constexpr (TWO_PHASE_FOLD) {
        auto do_fold_owned = [&](uint32_t owned) {
            cb_wait_front(cb_gather, owned * K);  // writer gathered owned rows × K partials
            cb_reserve_back(cb_rowpartial, owned);
            reconfig_data_format(cb_gather, cb_gather);
            pack_reconfig_data_format(cb_rowpartial);
            for (uint32_t l = 0; l < owned; ++l) {
                const uint32_t g = l * K;  // owned row l's K partials are contiguous [g, g+K)
                tile_regs_acquire();
                uint32_t first_pair = 0;
                if (K & 1u) {
                    copy_tile_to_dst_init_short(cb_gather);
                    copy_tile(cb_gather, g + 0, 0);
                    first_pair = 1;
                }
                if (K > 1) {
                    add_tiles_init(cb_gather, cb_gather, /*acc_to_dest=*/true);
                    for (uint32_t k = first_pair; k < K; k += 2) {
                        add_tiles(cb_gather, cb_gather, g + k, g + k + 1, 0);
                    }
                }
                binop_with_scalar_tile_init();
                add_unary_tile(0, eps_bits);
                rsqrt_tile_init();
                rsqrt_tile(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_rowpartial);  // finalized 1/RMS for owned row l
                tile_regs_release();
            }
            cb_push_back(cb_rowpartial, owned);
            cb_pop_front(cb_gather, owned * K);
        };
        // Refinement 6f lever 2: overlap the distributed round under compute (the complementary
        // step to R6e). R6e shipped this loop FULLY SYNCHRONOUS because R6b measured pipelining
        // flat while the master's serial fold sat ON the round's critical path. R6e distributed
        // that fold off the single master (folders own disjoint tile-rows), so re-enabling the
        // R6b lookahead now WINS: issue batch r+1's pass 1 BEFORE batch r's fold/pass 2 so the
        // local reduce overlaps the writer's synchronous cross-core round. cb_stat_local is 2*C
        // deep (two rounds in flight); the writer is serial across rounds so cb_gather /
        // cb_stat_global / cb_rowpartial stay one-round-in-flight; the semaphore/scatter protocol
        // and the fixed-base addressing are UNCHANGED — only the compute ISSUE order changes, so
        // the output is byte-identical to the R6e sequential loop. PIPELINE_LOOKAHEAD=0 or a
        // single-round group (num_rounds==1) degenerates to that sequential loop.
        if constexpr (PIPELINE_LOOKAHEAD) {
            if (num_rounds > 0) {
                do_pass1(0, c_of(0));  // prologue: batch 0's partials
            }
            for (uint32_t r = 0; r < num_rounds; ++r) {
                if (r + 1 < num_rounds) {
                    do_pass1((r + 1) * C_ROWS, c_of(r + 1));  // overlap batch r+1's pass 1 with round r
                }
                if (is_folder) {
                    do_fold_owned(owned_max);
                }
                do_pass2(r * C_ROWS, c_of(r));
            }
        } else {
            for (uint32_t r = 0; r < num_rounds; ++r) {
                do_pass1(r * C_ROWS, c_of(r));
                if (is_folder) {
                    do_fold_owned(owned_max);
                }
                do_pass2(r * C_ROWS, c_of(r));
            }
        }
        cb_pop_front(cb_x_in, shard_tiles);
        if constexpr (HAS_GAMMA) {
            cb_pop_front(cb_gamma, vwt);
        }
        cb_pop_front(cb_scaler, HAS_PARTIAL_W ? 2 : 1);
        return;
    }

    // Round loop. PIPELINE_LOOKAHEAD (R6b lever 2) selects the issue order:
    //  * OFF (byte-identical to R6a): strictly sequential per round — pass 1 -> fold -> pass 2.
    //  * ON (shipped on the tiled path): batch r+1's pass 1 is issued one round AHEAD so the
    //      local reduce overlaps the writer's synchronous cross-core round. Wins 1.05x on
    //      BLOCK 8x8 once lever 1 has compacted the gather (its complementary step). Both loops
    //      are numerically identical (same ops, same per-round order) — only the ISSUE order of
    //      the next round's pass 1 differs; single-round groups degenerate to the OFF loop.
    if constexpr (PIPELINE_LOOKAHEAD) {
        if (num_rounds > 0) {
            do_pass1(0, c_of(0));  // prologue: batch 0's partials
        }
        for (uint32_t r = 0; r < num_rounds; ++r) {
            if (r + 1 < num_rounds) {
                do_pass1((r + 1) * C_ROWS, c_of(r + 1));  // overlap batch r+1's pass 1 with round r
            }
            if (is_master) {
                do_fold(c_of(r));
            }
            do_pass2(r * C_ROWS, c_of(r));
        }
    } else {
        for (uint32_t r = 0; r < num_rounds; ++r) {
            do_pass1(r * C_ROWS, c_of(r));
            if (is_master) {
                do_fold(c_of(r));
            }
            do_pass2(r * C_ROWS, c_of(r));
        }
    }

    cb_pop_front(cb_x_in, shard_tiles);
    if constexpr (HAS_GAMMA) {
        cb_pop_front(cb_gamma, vwt);
    }
    cb_pop_front(cb_scaler, HAS_PARTIAL_W ? 2 : 1);
}
