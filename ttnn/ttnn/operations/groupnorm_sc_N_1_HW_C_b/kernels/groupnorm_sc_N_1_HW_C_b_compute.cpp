// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// groupnorm_sc_N_1_HW_C_b compute.
//
// Per block (op_design.md "Block schedule"):
//   accumulate_moments  : per chunk [RM: tilize] -> shifted-square chain (x - k)^2 -> reduce(x) & reduce((x-k)^2)
//                         with Accumulate (UnpackToDestFp32 accumulators), then one copy into the FPU-readable chan
//                         CBs that also applies the shift to the first moment: S1' = sum(x) - n_rows * k
//   collapse            : matmul(chan row [1 x W], map [W x 1]) -> 2 local partial tiles (row 0 = group lanes)
//   combine (root)      : matmul(ones [1 x R], gathered [R x 2]) -> [S1', S2'] replicated rows;
//                         finalize chain -> [rstd, mean'] (mean' = mean - k)
//   expand              : matmul<transpose>(stats, map) -> per-channel rstd / mean' (rows replicated);
//                         a = rstd*gamma, m = mean'*a, b' = m - beta
//   normalize           : per chunk y = (x - k)*a - b' (BinaryFpu Sub + DestReuse Mul + DestReuse Sub), resident
//                         chunks first
// Shift k (cb_pilot_rows, row 0 per channel): one sample per group, x[image row 0, first channel of the group],
// identical on every core of a unit, so the shifted sums combine with no correction. The variance is then
// E[(x-k)^2] - E[x-k]^2, whose cancellation scales with (mean-k)^2 ~ var instead of mean^2, and pass 2 never
// hands an operand of size |mean| to the 19-bit srcA/srcB registers: x - k is exact in DEST (Refinement 1).
// Every helper keeps its default INPUT_AND_OUTPUT data-format reconfig; the one boot is the matmul
// contract compute_kernel_hw_startup<SrcOrder::Reverse>.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/cb_api.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"

using namespace compute_kernel_lib;

namespace {
constexpr uint32_t cb_input_rm = 0;
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
constexpr uint32_t cb_pilot_rows = 22;  // shift k per channel (row 0), held for the block

constexpr uint32_t W = get_compile_time_arg_val(0);
constexpr uint32_t CHUNK_HT = get_compile_time_arg_val(1);
constexpr bool IS_RM = get_compile_time_arg_val(2) != 0;
constexpr bool HAS_GAMMA = get_compile_time_arg_val(3) != 0;
constexpr bool HAS_BETA = get_compile_time_arg_val(4) != 0;
constexpr uint32_t R = get_compile_time_arg_val(5);
constexpr uint32_t EXP_SBW = get_compile_time_arg_val(6);       // expand matmul out_subblock_w
constexpr uint32_t OUT_CB_PAGES = get_compile_time_arg_val(7);  // cb_output_tiles capacity (alias group B)
constexpr uint32_t T = CHUNK_HT * W;

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
// UnpackToDestFp32 accumulators (exact fp32 reload); publish_channel_moments() hands them to the FPU.
ALWI void accumulate_chunk(uint32_t j, uint32_t rows_valid) {
    constexpr uint32_t x_cb = cb_input_tiles;
    DataflowBuffer x_dfb(x_cb);
    DataflowBuffer xsq_dfb(cb_x_squared);
    DataflowBuffer scaler_dfb(cb_reduce_scaler);
    DataflowBuffer sum_acc_dfb(cb_sum_acc);
    DataflowBuffer sumsq_acc_dfb(cb_sumsq_acc);
    DataflowBuffer pilot_dfb(cb_pilot_rows);

    cb_wait_front(x_cb, T);

    // (x - k)^2 -> cb_x_squared (Float32), the sequential intermediate of the second reduce. x - k is exact
    // in fp32 DEST (both bf16); the square is an fp32 SFPU multiply.
    eltwise_chain(
        IterationShape::grid(rows_valid, W),
        BinaryFpu<
            BinaryFpuOp::Sub,
            input(x_cb, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block),
            input(cb_pilot_rows, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row)>{
            x_dfb, pilot_dfb},
        Square<>{},
        PackTile<output(cb_x_squared)>{xsq_dfb});
    push_pad(cb_x_squared, T - rows_valid * W);
    cb_wait_front(cb_x_squared, T);

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
        x_dfb, scaler_dfb, sum_acc_dfb, shape, ReduceInputMemoryLayout::contiguous(), Accumulate(sum_acc_dfb, j));
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
// UnpackToDestFp32 CB is exact; the first moment is shifted in fp32 here, S1' = sum(x) - n_rows * k (row 0;
// the collapse reads row 0 only). One copy per block replaces a last-chunk variant of both reduces, which
// would double the reduce instantiations (kernel text must fit the kernel-config ring).
ALWI void publish_channel_moments(uint32_t n_rows_bits) {
    DataflowBuffer sum_acc_dfb(cb_sum_acc);
    DataflowBuffer sumsq_acc_dfb(cb_sumsq_acc);
    DataflowBuffer pilot_dfb(cb_pilot_rows);
    DataflowBuffer chan_scale_dfb(cb_chan_scale);
    DataflowBuffer chan_work_dfb(cb_chan_work);
    eltwise_chain(
        IterationShape::tiles(W),
        CopyTile<input(cb_sum_acc), Dst::D0>{sum_acc_dfb},
        CopyTile<input(cb_pilot_rows, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block), Dst::D1>{
            pilot_dfb},
        MulUnary<Dst::D1>{n_rows_bits},
        SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
        CopyTile<input(cb_sumsq_acc), Dst::D2>{sumsq_acc_dfb},
        PackTile<output(cb_chan_scale), Dst::D0>{chan_scale_dfb},
        PackTile<output(cb_chan_work), Dst::D2>{chan_work_dfb});
}

// Pass-2 normalize of the chunk at the front of cb_input_keep: y = (x - k)*a - b' (k, a, b' row-mapped, held
// for the block).
ALWI void normalize_chunk(uint32_t rows_valid) {
    constexpr uint32_t x_cb = cb_input_keep;
    DataflowBuffer x_dfb(x_cb);
    DataflowBuffer pilot_dfb(cb_pilot_rows);
    DataflowBuffer scale_dfb(cb_chan_scale);
    DataflowBuffer bprime_dfb(cb_bprime);
    DataflowBuffer out_dfb(cb_output_tiles);

    cb_wait_front(x_cb, T);
    eltwise_chain(
        IterationShape::grid(rows_valid, W),
        BinaryFpu<
            BinaryFpuOp::Sub,
            input(x_cb, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block),
            input(cb_pilot_rows, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row)>{
            x_dfb, pilot_dfb},
        DestReuseBinary<
            BinaryFpuOp::Mul,
            input(cb_chan_scale, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
            DestReuseType::DEST_TO_SRCB>{scale_dfb},
        DestReuseBinary<
            BinaryFpuOp::Sub,
            input(cb_bprime, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row),
            DestReuseType::DEST_TO_SRCA>{bprime_dfb},
        PackTile<output(cb_output_tiles)>{out_dfb});
    push_pad(cb_output_tiles, T - rows_valid * W);
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
    const uint32_t is_root = get_arg_val<uint32_t>(arg++);
    const uint32_t inv_n_bits = get_arg_val<uint32_t>(arg++);
    const uint32_t eps_bits = get_arg_val<uint32_t>(arg++);
    const uint32_t n_rows_bits = get_arg_val<uint32_t>(arg++);  // float(rows_core * 32): elements per lane

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

    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_chan_scale, cb_group_map, cb_local_partials);

    // The map is resident for the whole kernel (matmul in1, NoWaitNoPop).
    cb_wait_front(cb_group_map, W);

    const uint32_t num_chunks = (rows_core + CHUNK_HT - 1) / CHUNK_HT;
    const uint32_t K = resident_chunks < num_chunks ? resident_chunks : num_chunks;
    const uint32_t first_resident = num_chunks - K;
    auto rows_valid_of = [&](uint32_t j) {
        const uint32_t rows_left = rows_core - j * CHUNK_HT;
        return rows_left < CHUNK_HT ? rows_left : CHUNK_HT;
    };

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        // ---------------- accumulate_moments (pass 1) ----------------
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
        publish_channel_moments(n_rows_bits);

        // ---------------- collapse_to_group_partials ----------------
        // [1 x W] per-channel row x [W x 1] map -> one tile; row 0 holds the G_core group lanes.
        matmul_with_map<false>(chan_scale_cb, map_cb, local_partials_cb, MatmulBlockShape::of(1, 1, 1, 1, W, 1));
        matmul_with_map<false>(chan_work_cb, map_cb, local_partials_cb, MatmulBlockShape::of(1, 1, 1, 1, W, 1));

        // RM stall shadow: tilize the first pass-2 re-read chunk before blocking on the combine.
        bool pretilized = false;
        if constexpr (IS_RM) {
            if (first_resident > 0) {
                tilize_chunk(rows_valid_of(0));
                pretilized = true;
            }
        }

        // ---------------- combine_group_stats (root) ----------------
        if (is_root) {
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

            // Finalize: mean = S1/n, var = max(S2/n - mean^2, 0), rstd = rsqrt(var + eps) -> [rstd, mean].
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
                MulUnary<Dst::D0>{inv_n_bits},
                MulUnary<Dst::D1>{inv_n_bits},
                MulBinary<Dst::D0, Dst::D0, Dst::D2>{},
                SubBinary<Dst::D1, Dst::D2, Dst::D1>{},
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
                    Dst::D0>{stats_out_dfb, 1});
            cb_push_back(cb_stats_out, 2);
            cb_pop_front(cb_group_sums, 2);
        }

        // ---------------- expand_stats_to_channels ----------------
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
        // m = mean'_ch * a -> cb_chan_shift (alias of cb_chan_work: tile i over tile i, in place).
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

        // ---------------- normalize_block (pass 2): resident chunks first, then re-reads ----------------
        // cb_input_keep already fronts the resident tail; the re-read head lands behind it.
        for (uint32_t jj = 0; jj < num_chunks; ++jj) {
            const bool reread = jj >= K;
            const uint32_t j = reread ? jj - K : first_resident + jj;
            const uint32_t rows_valid = rows_valid_of(j);
            if constexpr (IS_RM) {
                if (reread && !(pretilized && j == 0)) {
                    tilize_chunk(rows_valid);
                }
            }
            normalize_chunk(rows_valid);
            if (reread) {
                skip_chunk(cb_input_tiles);  // the pass-1 view also carried the re-read
            }
        }
        cb_pop_front(cb_chan_scale, W);
        cb_pop_front(cb_bprime, W);
        cb_pop_front(cb_pilot_rows, W);
    }
}
