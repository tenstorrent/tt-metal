// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// groupnorm_sc_N_1_HW_C_b compute.
//
// Per block (op_design.md "Block schedule"):
//   accumulate_moments  : per chunk [RM: tilize] -> chain x*(x - k) -> reduce(x) & reduce(x*(x - k)) with Accumulate
//                         (UnpackToDestFp32 accumulators; the Sigma x one starts from the reader's exact -n*k seed,
//                         so it ends at S1' = Sigma (x - k)), then one exact copy into the FPU-readable chan CBs
//   collapse            : matmul(chan row [1 x W], map [W x 1]) -> 2 local partial tiles (row 0 = group lanes)
//   combine (root)      : matmul(ones [1 x R], gathered [R x 2]) -> [S1', T2] replicated rows;
//                         finalize chain (+ the group shift k) -> [rstd, mean]
//   expand              : matmul<transpose>(stats, map) -> per-channel rstd / mean (rows replicated);
//                         a = rstd*gamma, m = mean*a, b' = m - beta
//   normalize           : per chunk y = a*x - b', resident chunks first
// Shift k (Refinement 1): one sample per group, x[image row 0, first channel of the group], identical on every
// core of a unit, so the shifted sums combine with no correction. With mean' = S1'/n and T2 = Sigma x*(x - k),
//   var = T2/n - mean'*(mean' + k) = E[(x - k)^2] - mean'^2,
// whose cancellation scales with |k|*|mean - k| ~ |mean|*std instead of mean^2 (the one-pass E[x^2] - mean^2
// it replaces), and the 19-bit srcA/srcB truncation hits moments of that reduced size.
// Every helper keeps its default INPUT_AND_OUTPUT data-format reconfig; the one boot is the matmul
// contract compute_kernel_hw_startup<SrcOrder::Reverse>.

// Debug builds only (watcher / LLK asserts, i.e. --dev): the asserts inflate TRISC text, and GCC's IPA
// constant-propagation clones of the per-chunk wrappers then overflow the 70 688 B kernel-config ring on the RM
// root binary (measured up to 3.5 KB over). Production builds keep the clones: measured faster (sharded TILE
// vae_16384x512 70.0 us vs 78.0 us without them).
#if defined(WATCHER_ENABLED) || defined(WATCHER_NOINLINE) || defined(ENABLE_LLK_ASSERT)
#pragma GCC optimize("no-ipa-cp-clone")
#endif
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"

using namespace compute_kernel_lib;

namespace {
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_output_rm = 1;  // sharded RM: the output shard (untilized rows)
// The input ring: one allocation, two FIFO views pushed in lockstep (alias group A). Pass 1 consumes every
// chunk through cb_input_tiles; cb_input_keep drops the streamed head chunks during pass 1 and holds the
// resident tail for pass 2, followed by the re-read head - exactly the pass-2 order.
constexpr uint32_t cb_input_tiles = 2;
constexpr uint32_t cb_input_keep = 3;
constexpr uint32_t cb_x_squared = 4;
constexpr uint32_t cb_output_tiles = 5;
constexpr uint32_t cb_reduce_scaler = 6;
constexpr uint32_t cb_sum_acc = 7;
constexpr uint32_t cb_sumsq_acc = 8;
constexpr uint32_t cb_chan_scale = 9;
constexpr uint32_t cb_chan_work = 10;
constexpr uint32_t cb_chan_shift = 11;
constexpr uint32_t cb_group_map = 12;
constexpr uint32_t cb_gamma_rows = 13;
constexpr uint32_t cb_beta_rows = 14;
constexpr uint32_t cb_local_partials = 15;
constexpr uint32_t cb_gathered_partials = 16;
constexpr uint32_t cb_ones_tiles = 17;
constexpr uint32_t cb_group_sums = 18;
constexpr uint32_t cb_stats_out = 19;
constexpr uint32_t cb_group_stats = 20;
constexpr uint32_t cb_pilot_rows = 22;   // shift k per channel (row 0), held for pass 1
constexpr uint32_t cb_sum_seed = 23;     // -n*k per channel (row 0), the Sigma x accumulator's start
constexpr uint32_t cb_group_shift = 24;  // root: k per group lane, rows replicated
constexpr uint32_t cb_xk = 26;           // P1L: bf16 ring of x - k (two DEST windows)

constexpr uint32_t W = get_compile_time_arg_val(0);
constexpr uint32_t CHUNK_HT = get_compile_time_arg_val(1);
constexpr bool IS_RM = get_compile_time_arg_val(2) != 0;
constexpr bool HAS_GAMMA = get_compile_time_arg_val(3) != 0;
constexpr bool HAS_BETA = get_compile_time_arg_val(4) != 0;
constexpr uint32_t R = get_compile_time_arg_val(5);
constexpr uint32_t EXP_SBW = get_compile_time_arg_val(6);       // expand matmul out_subblock_w
constexpr uint32_t OUT_CB_PAGES = get_compile_time_arg_val(7);  // cb_output_tiles capacity (alias group B)
// The input / output are L1 shards backing their CBs (TILE: the input ring and cb_output_tiles; RM: cb_input_rm
// and cb_output_rm). Sharded RM untilizes each normalized chunk into the output shard.
constexpr bool SHARDED = get_compile_time_arg_val(8) != 0;
constexpr bool UNTILIZE_OUT = SHARDED && IS_RM;
// One binary per role (the unit root combines and finalizes). Compile-time so the block loop is not unswitched on
// it (two copies of the whole per-block body would not fit the kernel-config ring under the watcher build).
constexpr bool IS_ROOT = get_compile_time_arg_val(9) != 0;
// m copies of the W pilot tiles (host pilot_repeat, m | CHUNK_HT): pass 1's shift chain walks m tile rows as one
// row of m*W tiles, so its DEST window fills on narrow slices too (the window only spans one walk row).
constexpr uint32_t PILOT_REPEAT = get_compile_time_arg_val(10);
constexpr uint32_t PILOT_TILES = PILOT_REPEAT * W;
// Per-chunk device zones (perf_instrumentation.hpp): the host enables them only when every core's chunk count keeps
// the zone executions inside the per-RISC marker budget (a long DRAM block would silently exhaust it).
constexpr bool CHUNK_ZONES = get_compile_time_arg_val(11) != 0;
constexpr uint32_t T = CHUNK_HT * W;
// Tiles per DEST window of the two-op per-chunk chains (pass-1 x*(x - k), pass-2 a*x - b'): each element runs
// over the whole window, so its FPU init is paid once per window, not per tile. The chain clamps it to DEST.
constexpr uint32_t CHAIN_BLOCK = DEST_AUTO_LIMIT;

// b' lives in cb_chan_work when beta is present (5e writes it back in place), else in cb_chan_shift.
constexpr uint32_t cb_bprime = HAS_BETA ? cb_chan_work : cb_chan_shift;

// RM tilize is called from three sites (pass 1 / pre-tilize / pass 2) with a runtime extent; it is emitted
// once rather than inlined per site, so the five kernel binaries fit the kernel-config ring also under the
// watcher/assert (--dev) build. (The map matmuls stay inlined: measured, an out-of-line body grows the code.)
#define NOINLINE_BLOCK_OP __attribute__((noinline))

// Push (and for a CB-backed producer, reserve) the nominal-T pad of a ragged chunk.
ALWI void push_pad(uint32_t cb, uint32_t pad) {
    if (pad > 0) {
        cb_reserve_back(cb, pad);
        cb_push_back(cb, pad);
    }
}

// RM: tilize one chunk (rows_valid tile-rows of W tiles) from cb_input_rm into the input ring, then consume
// the rm pad pages and push the pad so every ring advances by the nominal T (both ring views in lockstep).
NOINLINE_BLOCK_OP void tilize_chunk(uint32_t rows_valid) {
    MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_tilize");
    DataflowBuffer rm_dfb(cb_input_rm);
    DataflowBuffer x_dfb(cb_input_tiles);
    cb_reserve_back(cb_input_keep, T);
    tilize<W, cb_input_rm, cb_input_tiles>(rm_dfb, x_dfb, rows_valid);
    const uint32_t pad = T - rows_valid * W;
    if (pad > 0) {
        cb_wait_front(cb_input_rm, pad);
        cb_pop_front(cb_input_rm, pad);
        push_pad(cb_input_tiles, pad);
    }
    cb_push_back(cb_input_keep, T);
}

// Advance one view of the input ring past a chunk it does not read.
ALWI void skip_chunk(uint32_t cb) {
    cb_wait_front(cb, T);
    cb_pop_front(cb, T);
}

// Pass-1 statistics of one chunk at the front of cb_input_tiles. Both moments accumulate into the
// UnpackToDestFp32 accumulators (exact fp32 reload); chunk 0 of Sigma x reloads the -n*k seed instead, so the
// accumulator ends at S1' = Sigma (x - k). publish_channel_moments() hands them to the FPU.
ALWI void accumulate_chunk(uint32_t j, uint32_t rows_valid) {
    constexpr uint32_t x_cb = cb_input_tiles;
    DataflowBuffer x_dfb(x_cb);
    DataflowBuffer xsq_dfb(cb_x_squared);
    DataflowBuffer scaler_dfb(cb_reduce_scaler);
    DataflowBuffer sum_acc_dfb(cb_sum_acc);
    DataflowBuffer sumsq_acc_dfb(cb_sumsq_acc);
    DataflowBuffer pilot_dfb(cb_pilot_rows);
    DataflowBuffer seed_dfb(cb_sum_seed);

    {
        MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_p1_wait_x");
        cb_wait_front(x_cb, T);
    }

    // x*(x - k) -> cb_x_squared (Float32), the sequential intermediate of the second reduce. x - k (fp32 DEST)
    // re-enters the FPU through the bf16 srcB the pilot left: exact whenever k/2 <= x <= 2k (Sterbenz), the
    // offset-heavy regime the shift exists for; elsewhere |x - k| ~ std and its rounding is ordinary noise.
    // Walk ceil(rows_valid/m) rows of m*W tiles (m | CHUNK_HT keeps a ragged chunk's rounded-up walk inside its
    // nominal T pages; the extra tiles are pad, never reduced).
    const uint32_t walk_rows = (rows_valid + PILOT_REPEAT - 1) / PILOT_REPEAT;
    {
        MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_p1_xk_chain");
        // perf_experiments/per_tile_pass_chains P1L: x - k leaves through the packer into a bf16 ring (cb_xk) and
        // re-enters through the unpacker, instead of the dest-reuse MOVD2B (which moves DEST's hi16, i.e. a bf16
        // truncation, and stalls per face). Windows are software-pipelined one deep (sub w+1 before mul w) so the
        // unpacker never waits on the pack of the window it is about to multiply.
        const uint32_t n_tiles = walk_rows * PILOT_TILES;
        auto window_len = [&](uint32_t t) {
            const uint32_t left_in_row = PILOT_TILES - t % PILOT_TILES;
            const uint32_t c = left_in_row < CHAIN_BLOCK ? left_in_row : CHAIN_BLOCK;
            return c;
        };
        auto sub_window = [&](uint32_t t, uint32_t cnt) {
            sub_bcast_rows_init(x_cb, cb_pilot_rows);
            cb_reserve_back(cb_xk, cnt);
            tile_regs_acquire();
            for (uint32_t j = 0; j < cnt; ++j) {
                sub_tiles_bcast_rows(x_cb, cb_pilot_rows, t + j, (t + j) % PILOT_TILES, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_xk);
            for (uint32_t j = 0; j < cnt; ++j) {
                pack_tile(j, cb_xk);
            }
            tile_regs_release();
            cb_push_back(cb_xk, cnt);
        };
        auto mul_window = [&](uint32_t t, uint32_t cnt) {
            mul_init(x_cb, cb_xk);
            cb_wait_front(cb_xk, cnt);
            cb_reserve_back(cb_x_squared, cnt);
            tile_regs_acquire();
            for (uint32_t j = 0; j < cnt; ++j) {
                mul_tiles(x_cb, cb_xk, t + j, j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_x_squared);
            for (uint32_t j = 0; j < cnt; ++j) {
                pack_tile(j, cb_x_squared);
            }
            tile_regs_release();
            cb_push_back(cb_x_squared, cnt);
            cb_pop_front(cb_xk, cnt);
        };
        cb_wait_front(cb_pilot_rows, PILOT_TILES);
        reconfig_data_format(x_cb, cb_pilot_rows);  // x, pilot and the x - k ring are all bf16: one srcA/srcB format
        uint32_t prev_t = 0;
        uint32_t prev_cnt = window_len(0);
        sub_window(0, prev_cnt);
        for (uint32_t t = prev_cnt; t < n_tiles;) {
            const uint32_t cnt = window_len(t);
            sub_window(t, cnt);
            mul_window(prev_t, prev_cnt);
            prev_t = t;
            prev_cnt = cnt;
            t += cnt;
        }
        mul_window(prev_t, prev_cnt);
        push_pad(cb_x_squared, T - n_tiles);
        cb_wait_front(cb_x_squared, T);
    }

    MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_p1_reduce");
    const auto shape = ReduceInputBlockShape::of(rows_valid, W);
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_COL,
        x_cb,
        cb_reduce_scaler,
        cb_sum_acc,
        ReduceInputPolicy::NoWaitNoPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::ReduceTile>(
        x_dfb,
        scaler_dfb,
        sum_acc_dfb,
        shape,
        ReduceInputMemoryLayout::contiguous(),
        Accumulate(j == 0 ? seed_dfb : sum_acc_dfb, j == 0 ? 1 : j));
    reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_COL,
        cb_x_squared,
        cb_reduce_scaler,
        cb_sumsq_acc,
        ReduceInputPolicy::NoWaitNoPop,
        ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ReduceAlgorithm::ReduceTile>(
        xsq_dfb, scaler_dfb, sumsq_acc_dfb, shape, ReduceInputMemoryLayout::contiguous(), Accumulate(sumsq_acc_dfb, j));

    cb_pop_front(x_cb, T);
    cb_pop_front(cb_x_squared, T);
}

// Per-channel moments [1 x W] -> the FPU-readable chan CBs (matmul in0 of the collapse). CopyTile from an
// UnpackToDestFp32 CB is exact. One copy per block replaces a last-chunk variant of both reduces, which would
// double the reduce instantiations (kernel text must fit the kernel-config ring).
ALWI void publish_channel_moments() {
    DataflowBuffer sum_acc_dfb(cb_sum_acc);
    DataflowBuffer sumsq_acc_dfb(cb_sumsq_acc);
    DataflowBuffer chan_scale_dfb(cb_chan_scale);
    DataflowBuffer chan_work_dfb(cb_chan_work);
    eltwise_chain(
        IterationShape::tiles(W),
        CopyTile<input(cb_sum_acc), Dst::D0>{sum_acc_dfb},
        CopyTile<input(cb_sumsq_acc), Dst::D1>{sumsq_acc_dfb},
        PackTile<output(cb_chan_scale), Dst::D0>{chan_scale_dfb},
        PackTile<output(cb_chan_work), Dst::D1>{chan_work_dfb});
}

// Pass-2 normalize of the chunk at the front of cb_input_keep: y = a*x - b' (a, b' row-mapped, held for the
// block), as two accumulating FPU multiplies into one DEST tile: DEST = (-1)*b' + x*a.
// perf_experiments/per_tile_pass_chains (N1b): raw compute API instead of eltwise_chain(BinaryFpu Mul +
// DestReuseBinary Sub). Mechanism: ELWMUL always accumulates into DEST, so the subtraction is folded into a
// second multiply by a constant -1 tile; both multiplies share one (srcA bf16, srcB Float32) format pair and one
// MOP, so there is no per-window re-init / reconfig and no DEST->src move (the dest-reuse MOVD2A stalls per face).
// Precision: b' and a sit on srcB (Float32 -> tf32, every bit used under HiFi3); -1 and x are bf16 on srcA (exact);
// the sum is formed in fp32 DEST (the dest-reuse path truncated a*x to tf32 on the DEST -> srcA move).
constexpr uint32_t cb_neg_one = 25;
ALWI void normalize_chunk(uint32_t rows_valid) {
    constexpr uint32_t x_cb = cb_input_keep;
    {
        MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_p2_wait_x");
        cb_wait_front(x_cb, T);
    }
    MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_p2_normalize");
    cb_wait_front(cb_chan_scale, W);
    cb_wait_front(cb_bprime, W);
    const uint32_t n = rows_valid * W;
    reconfig_data_format(cb_neg_one, cb_bprime);
    pack_reconfig_data_format(cb_output_tiles);
    mul_init(cb_neg_one, cb_bprime);
    uint32_t w = 0;
    for (uint32_t i = 0; i < n; i += CHAIN_BLOCK) {
        const uint32_t cnt = n - i < CHAIN_BLOCK ? n - i : CHAIN_BLOCK;
        cb_reserve_back(cb_output_tiles, cnt);
        tile_regs_acquire();
        uint32_t ww = w;
        for (uint32_t j = 0; j < cnt; ++j) {
            mul_tiles(cb_neg_one, cb_bprime, 0, ww, j);
            ww = ww + 1 == W ? 0 : ww + 1;
        }
        ww = w;
        for (uint32_t j = 0; j < cnt; ++j) {
            mul_tiles(x_cb, cb_chan_scale, i + j, ww, j);
            ww = ww + 1 == W ? 0 : ww + 1;
        }
        w = ww;
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < cnt; ++j) {
            pack_tile(j, cb_output_tiles);
        }
        tile_regs_release();
        cb_push_back(cb_output_tiles, cnt);
    }
    push_pad(cb_output_tiles, T - n);
    cb_pop_front(x_cb, T);
}

// Sharded RM (UNTILIZE_OUT): the same folded normalize, packed straight to the row-major output shard with
// pack_untilize_dest (perf_experiments/per_tile_pass_chains N1u). Bypasses PackTile -> cb_output_tiles ->
// untilize<>() (unpack + datacopy + pack again per tile): the normalized DEST block is untilized by the packer.
// DEST block = UB_RT tile rows x UB_CT tile columns (UB_CT | W, UB_CT * UB_RT <= DEST window); the packer writes
// row r of the block at r * W pages and column block b at b * UB_CT tiles of the reserved tile rows.
constexpr uint32_t largest_divisor_le(uint32_t n, uint32_t m) {
    for (uint32_t b = m; b > 1; --b) {
        if (n % b == 0) {
            return b;
        }
    }
    return 1;
}
constexpr uint32_t UB_CT = largest_divisor_le(W, CHAIN_BLOCK);
constexpr uint32_t UB_RT = CHAIN_BLOCK / UB_CT;
constexpr uint32_t UB_NB = W / UB_CT;

ALWI void normalize_untilize_chunk(uint32_t rows_valid) {
    constexpr uint32_t x_cb = cb_input_keep;
    {
        MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_p2_wait_x");
        cb_wait_front(x_cb, T);
    }
    MaybeDeviceZoneScopeIf(CHUNK_ZONES, "c_p2_normalize");
    cb_wait_front(cb_chan_scale, W);
    cb_wait_front(cb_bprime, W);
    reconfig_data_format(cb_neg_one, cb_bprime);
    mul_init(cb_neg_one, cb_bprime);
    pack_untilize_dest_init<UB_CT, W>(cb_output_rm);
    for (uint32_t r0 = 0; r0 < rows_valid; r0 += UB_RT) {
        const uint32_t rt = rows_valid - r0 < UB_RT ? rows_valid - r0 : UB_RT;
        cb_reserve_back(cb_output_rm, rt * W);
        for (uint32_t b = 0; b < UB_NB; ++b) {
            const uint32_t c0 = b * UB_CT;
            tile_regs_acquire();
            for (uint32_t r = 0; r < rt; ++r) {
                for (uint32_t c = 0; c < UB_CT; ++c) {
                    mul_tiles(cb_neg_one, cb_bprime, 0, c0 + c, r * UB_CT + c);
                }
            }
            for (uint32_t r = 0; r < rt; ++r) {
                for (uint32_t c = 0; c < UB_CT; ++c) {
                    mul_tiles(x_cb, cb_chan_scale, (r0 + r) * W + c0 + c, c0 + c, r * UB_CT + c);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<UB_CT, W>(cb_output_rm, rt, b);
            tile_regs_release();
        }
        cb_push_back(cb_output_rm, rt * W);
    }
    pack_untilize_uninit(cb_output_rm);
    cb_pop_front(x_cb, T);
}

// matmul_block with the map as in1 (resident, never popped): collapse (transpose=false) / expand (true).
template <bool transpose>
ALWI void matmul_with_map(
    CircularBuffer& in0, CircularBuffer& map, CircularBuffer& out, const MatmulBlockShape& shape) {
    matmul_block<
        transpose,
        false,
        LastBlockTarget::Out,
        OutputCBLayout::SubblockMajor,
        matmul_config::InitMode::Short,
        InputPolicy::WaitAndPopPerKBlock,
        InputPolicy::NoWaitNoPop>(in0, map, out, out, shape);
}

}  // namespace

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t num_blocks = get_arg_val<uint32_t>(arg++);
    const uint32_t rows_core = get_arg_val<uint32_t>(arg++);
    const uint32_t resident_chunks = get_arg_val<uint32_t>(arg++);
    const uint32_t inv_n_bits = get_arg_val<uint32_t>(arg++);
    const uint32_t eps_bits = get_arg_val<uint32_t>(arg++);

    CircularBuffer chan_scale_cb(cb_chan_scale);
    CircularBuffer chan_work_cb(cb_chan_work);
    CircularBuffer map_cb(cb_group_map);
    CircularBuffer local_partials_cb(cb_local_partials);
    CircularBuffer ones_cb(cb_ones_tiles);
    CircularBuffer gathered_cb(cb_gathered_partials);
    CircularBuffer group_sums_cb(cb_group_sums);
    CircularBuffer group_stats_cb(cb_group_stats);

    DataflowBuffer chan_scale_dfb(cb_chan_scale);
    DataflowBuffer chan_work_dfb(cb_chan_work);
    DataflowBuffer chan_shift_dfb(cb_chan_shift);
    DataflowBuffer gamma_dfb(cb_gamma_rows);
    DataflowBuffer beta_dfb(cb_beta_rows);
    DataflowBuffer group_sums_dfb(cb_group_sums);
    DataflowBuffer stats_out_dfb(cb_stats_out);
    DataflowBuffer group_shift_dfb(cb_group_shift);

    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_chan_scale, cb_group_map, cb_local_partials);

    // N1b: the constant -1 tile (bf16, srcA of the folded subtraction), resident for the whole kernel.
    {
        DataflowBuffer neg_one_dfb(cb_neg_one);
        eltwise_chain(
            IterationShape::one_tile(), FillScalar<Dst::D0>{-1.0f}, PackTile<output(cb_neg_one)>{neg_one_dfb});
    }

    // The map is resident for the whole kernel (matmul in1, NoWaitNoPop).
    {
        MaybeDeviceZoneScope("c_wait_map");
        cb_wait_front(cb_group_map, W);
    }

    const uint32_t num_chunks = (rows_core + CHUNK_HT - 1) / CHUNK_HT;
    const uint32_t K = resident_chunks < num_chunks ? resident_chunks : num_chunks;
    const uint32_t first_resident = num_chunks - K;
    auto rows_valid_of = [&](uint32_t j) {
        const uint32_t rows_left = rows_core - j * CHUNK_HT;
        return rows_left < CHUNK_HT ? rows_left : CHUNK_HT;
    };

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        // ---------------- accumulate_moments (pass 1) ----------------
        {
            MaybeDeviceZoneScope("c_pass1");
            for (uint32_t j = 0; j < num_chunks; ++j) {
                const uint32_t rows_valid = rows_valid_of(j);
                if constexpr (IS_RM) {
                    tilize_chunk(rows_valid);
                }
                if (blk > 0 && j == 0) {
                    // Alias group B: the writer must have drained the previous block's output before x^2 lands.
                    cb_reserve_back(cb_output_tiles, OUT_CB_PAGES);
                }
                accumulate_chunk(j, rows_valid);
                if (j < first_resident) {
                    skip_chunk(cb_input_keep);  // streamed head chunk: not held for pass 2
                }
            }
            cb_pop_front(cb_pilot_rows, PILOT_TILES);
        }
        {
            MaybeDeviceZoneScope("c_collapse");
            publish_channel_moments();

            // ---------------- collapse_to_group_partials ----------------
            // [1 x W] per-channel row x [W x 1] map -> one tile; row 0 holds the G_core group lanes.
            matmul_with_map<false>(chan_scale_cb, map_cb, local_partials_cb, MatmulBlockShape::of(1, 1, 1, 1, W, 1));
            matmul_with_map<false>(chan_work_cb, map_cb, local_partials_cb, MatmulBlockShape::of(1, 1, 1, 1, W, 1));
        }

        // RM stall shadow: tilize the first pass-2 re-read chunk before blocking on the combine.
        bool pretilized = false;
        if constexpr (IS_RM) {
            if (first_resident > 0) {
                tilize_chunk(rows_valid_of(0));
                pretilized = true;
            }
        }

        // ---------------- combine_group_stats (root) ----------------
        if constexpr (IS_ROOT) {
            {
                // The gather round trip: every unit core's partial row landed (writer semaphore wait).
                MaybeDeviceZoneScope("c_wait_gather");
                cb_wait_front(cb_gathered_partials, 2 * R);
            }
            MaybeDeviceZoneScope("c_finalize");
            // Sum the cpi sender rows: ones [1 x R] x gathered [R x 2] -> [S1, S2], every row = the total.
            matmul_block<
                false,
                false,
                LastBlockTarget::Out,
                OutputCBLayout::SubblockMajor,
                matmul_config::InitMode::Short,
                InputPolicy::WaitAndPopPerKBlock,
                InputPolicy::WaitAndPopPerKBlock>(
                ones_cb, gathered_cb, group_sums_cb, group_sums_cb, MatmulBlockShape::of(1, 1, 1, 2, R, 1));

            // Finalize: mean' = S1'/n, mean = mean' + k, var = max(T2/n - mean'*mean, 0), rstd = rsqrt(var + eps)
            // -> [rstd, mean].
            cb_wait_front(cb_group_sums, 2);
            cb_reserve_back(cb_stats_out, 2);
            eltwise_chain(
                IterationShape::one_tile(),
                CopyTile<
                    input(
                        cb_group_sums,
                        WaitPolicy::None,
                        PopPolicy::None,
                        InputTileMapping::Scalar,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Offset),
                    Dst::D0>{group_sums_dfb, 0},
                CopyTile<
                    input(
                        cb_group_sums,
                        WaitPolicy::None,
                        PopPolicy::None,
                        InputTileMapping::Scalar,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Offset),
                    Dst::D1>{group_sums_dfb, 1},
                CopyTile<input(cb_group_shift), Dst::D2>{group_shift_dfb},
                MulUnary<Dst::D0>{inv_n_bits},
                MulUnary<Dst::D1>{inv_n_bits},
                AddBinary<Dst::D2, Dst::D0, Dst::D2>{},
                MulBinary<Dst::D0, Dst::D2, Dst::D0>{},
                SubBinary<Dst::D1, Dst::D0, Dst::D1>{},
                Relu<Dst::D1>{},
                AddUnary<Dst::D1>{eps_bits},
                Rsqrt<Approx::Exact, Dst::D1>{},
                PackTile<
                    output(
                        cb_stats_out,
                        ReservePolicy::None,
                        PushPolicy::None,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Offset),
                    Dst::D1>{stats_out_dfb, 0},
                PackTile<
                    output(
                        cb_stats_out,
                        ReservePolicy::None,
                        PushPolicy::None,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Offset),
                    Dst::D2>{stats_out_dfb, 1});
            cb_push_back(cb_stats_out, 2);
            cb_pop_front(cb_group_sums, 2);
        }

        // ---------------- expand_stats_to_channels ----------------
        {
            // The combine round trip seen from this core: [rstd, mean] multicast back from the unit root.
            MaybeDeviceZoneScope("c_wait_stats");
            cb_wait_front(cb_group_stats, 2);
        }
        {
            MaybeDeviceZoneScope("c_expand");
            // stats tile [32 x 32 lanes] x map^T [lanes x 32 channels] -> per-channel value, rows replicated.
            const auto expand_shape = MatmulBlockShape::of(1, W / EXP_SBW, 1, EXP_SBW, 1, 1);
            if constexpr (HAS_GAMMA) {
                // Both expands land in cb_chan_work (rstd_ch, then mean_ch): one call site, two iterations.
                for (uint32_t s = 0; s < 2; ++s) {
                    matmul_with_map<true>(group_stats_cb, map_cb, chan_work_cb, expand_shape);
                    if (s == 0) {
                        eltwise_chain(
                            IterationShape::tiles(W),
                            BinaryFpu<BinaryFpuOp::Mul, input(cb_chan_work), input(cb_gamma_rows, BroadcastDim::Row)>{
                                chan_work_dfb, gamma_dfb},
                            PackTile<output(cb_chan_scale)>{chan_scale_dfb});  // a = rstd_ch * gamma
                    }
                }
            } else {
                matmul_with_map<true>(group_stats_cb, map_cb, chan_scale_cb, expand_shape);  // a = rstd_ch
                matmul_with_map<true>(group_stats_cb, map_cb, chan_work_cb, expand_shape);   // mean_ch
            }
            // m = mean_ch * a -> cb_chan_shift (alias of cb_chan_work: tile i over tile i, in place).
            eltwise_chain(
                IterationShape::tiles(W),
                BinaryFpu<
                    BinaryFpuOp::Mul,
                    input(cb_chan_work),
                    input(cb_chan_scale, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block)>{
                    chan_work_dfb, chan_scale_dfb},
                PackTile<output(cb_chan_shift)>{chan_shift_dfb});
            if constexpr (HAS_BETA) {
                // b' = m - beta -> cb_chan_work (in place over cb_chan_shift's bytes).
                eltwise_chain(
                    IterationShape::tiles(W),
                    BinaryFpu<BinaryFpuOp::Sub, input(cb_chan_shift), input(cb_beta_rows, BroadcastDim::Row)>{
                        chan_shift_dfb, beta_dfb},
                    PackTile<output(cb_chan_work)>{chan_work_dfb});
            }
        }

        // ---------------- normalize_block (pass 2): resident chunks first, then re-reads ----------------
        // cb_input_keep already fronts the resident tail; the re-read head lands behind it.
        // In place (sharded, the output CBs are the input shard's bytes) needs no extra ordering: TILE packs tile i
        // over the tile i it just unpacked (unpack -> math -> pack per tile); RM untilizes a chunk over the rows it
        // was tilized from, which happened in pass 1 (resident) or in the tilize_chunk just above (re-read).
        MaybeDeviceZoneScope("c_pass2");
        if constexpr (UNTILIZE_OUT) {
            // The output shard is a FIFO in row order: skip the head region (re-read last), write the resident
            // tail, then wrap to the head. The writer drains every credit.
            push_pad(cb_output_rm, first_resident * T);
        }
        for (uint32_t jj = 0; jj < num_chunks; ++jj) {
            const bool reread = jj >= K;
            const uint32_t j = reread ? jj - K : first_resident + jj;
            const uint32_t rows_valid = rows_valid_of(j);
            if constexpr (IS_RM) {
                if (reread && !(pretilized && j == 0)) {
                    tilize_chunk(rows_valid);
                }
            }
            if constexpr (UNTILIZE_OUT) {
                normalize_untilize_chunk(rows_valid);
            } else {
                normalize_chunk(rows_valid);
            }
            if (reread) {
                skip_chunk(cb_input_tiles);  // the pass-1 view also carried the re-read
            }
        }
        cb_pop_front(cb_chan_scale, W);
        cb_pop_front(cb_bprime, W);
    }
}
