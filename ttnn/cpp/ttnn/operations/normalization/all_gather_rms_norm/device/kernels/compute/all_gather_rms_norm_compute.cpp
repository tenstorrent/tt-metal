// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Single fused compute kernel for all_gather_rms_norm.
//
// A Tensix core runs exactly one compute kernel, so the whole fused pipeline lives here.
//
// ring_size == 1 (single device, no fabric): one fused pass per tile-row,
//   x^2 -> E[x^2] -> 1/sqrt(E[x^2]+eps) -> x*recip*gamma+beta. The input row (Wt tiles) is read once
//   into cb_inp and kept resident (used for both x^2 and the normalize step), then popped per row.
//
// ring_size > 1 (multi device): a chunked TWO-PASS scheme so the cross-device stats all-gather is
//   batched instead of one fabric barrier per row. For each chunk of up to `gather_chunk` rows:
//     pass 1: for each row, x^2 -> AVG-reduce (1/total_W) -> per-device partial into cb_local_stats;
//     [the writer batches the fabric all-gather of the whole chunk -> cb_gathered_stats]
//     pass 2: for each row, SUM the ring_size gathered partials -> global E[x^2] -> rsqrt -> normalize.
//   The reader re-streams the chunk's input for pass 2 (input is NOT kept resident across the gather).
//
// Math ported from the generic distributed kernels:
//   rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp  (E[x^2])
//   rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather.cpp (normalize + gamma/beta)

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

// (AGRMS_TRACE_LOCAL disabled -- #define it to re-enable the cb tile-dump traces below)
#ifdef AGRMS_TRACE_LOCAL
#include "api/debug/dprint_pages.h"
#endif

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

// x^2 over the resident Wt-tile input row, AVG-reduced (scaler = 1/reduce_factor) into a single tile in
// cb_dst. For ring_size == 1, reduce_factor == Wt so cb_dst is the global mean E[x^2]; for ring_size > 1,
// reduce_factor == total_W so cb_dst is the per-device partial sum(x^2_local)/total_W. `base` is the tile
// offset of this row within cb_inp (0 unless the caller keeps a whole chunk resident and indexes rows at
// base = r*Wt without popping between rows).
//
// Accumulation order MIRRORS wan's rmsnorm_pre_allgather.cpp exactly: each column-tile's x^2 is
// L1-accumulated element-wise onto a SINGLE intermediate tile (cb_x2[0]) -- i.e. sum-across-tiles FIRST --
// then one REDUCE_ROW sums the 32 columns. Float add is non-associative, so this ordering (not the
// reduce-library's per-tile-then-across-tiles order) is what makes the bf16 output bit-identical to wan.
ALWI void reduce_x2(
    uint32_t cb_inp,
    uint32_t cb_x2,
    uint32_t cb_reduce,
    uint32_t cb_dst,
    uint32_t Wt,
    uint32_t blk,
    uint32_t base = 0) {
    reconfig_data_format(cb_inp, cb_inp);
    pack_reconfig_data_format(cb_x2);
    PACK((llk_pack_reconfig_l1_acc(0)));  // overwrite (not accumulate) when starting this row
    mul_tiles_init(cb_inp, cb_inp);
    cb_reserve_back(cb_x2, 1);
    for (uint32_t wt = 0; wt < Wt; wt += blk) {
        tile_regs_acquire();
        for (uint32_t wtr = 0; wtr < blk; wtr++) {
            mul_tiles(cb_inp, cb_inp, base + wt + wtr, base + wt + wtr, wtr);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t wtr = 0; wtr < blk; wtr++) {
            // accumulate every x^2 column-tile onto the single intermediate tile cb_x2[0]
            pack_tile<true>(wtr, cb_x2, 0);
            if (wt == 0 && wtr == 0) {
                PACK((llk_pack_reconfig_l1_acc(1)));  // enable L1 accumulation after the first packed tile
            }
        }
        tile_regs_release();
    }
    cb_push_back(cb_x2, 1);
    PACK((llk_pack_reconfig_l1_acc(0)));  // disable L1 accumulation

    compute_kernel_lib::
        reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
            cb_x2, cb_reduce, cb_dst, compute_kernel_lib::ReduceInputBlockShape::single());
}

// 1/sqrt(E[x^2] + eps) then the fused per-block normalize: x * recip (-> * gamma) (-> + beta) -> cb_out.
// Expects cb_var (1 tile) and the input row cb_inp (Wt tiles) already waited-front; consumes cb_var and
// leaves cb_inp in place (the caller pops it).
template <bool do_gamma, bool do_beta>
ALWI void finalize_row(
    uint32_t cb_var,
    uint32_t cb_eps,
    uint32_t cb_recip_sqrt_var,
    uint32_t cb_inp,
    uint32_t cb_gamma,
    uint32_t cb_beta,
    uint32_t cb_x_normed,
    uint32_t cb_gamma_out,
    uint32_t cb_out,
    uint32_t Wt,
    uint32_t blk) {
    constexpr bool LEGACY_RSQRT = false;

    // --- 1/sqrt(E[x^2] + eps) ---
    cb_wait_front(cb_var, 1);
    cb_reserve_back(cb_recip_sqrt_var, 1);
    reconfig_data_format(cb_var, cb_eps);
    pack_reconfig_data_format(cb_recip_sqrt_var);
    add_tiles_init(cb_var, cb_eps);
    ACQ();
    add_tiles(cb_var, cb_eps, 0, 0, 0);
    rsqrt_tile_init<LEGACY_RSQRT>();
    rsqrt_tile<LEGACY_RSQRT>(0);
    pack_tile(0, cb_recip_sqrt_var);
    REL();
    cb_push_back(cb_recip_sqrt_var, 1);
    cb_pop_front(cb_var, 1);

    // --- normalize (+ optional gamma / beta), fused per block ---
    cb_wait_front(cb_recip_sqrt_var, 1);
    for (uint32_t wt = 0; wt < Wt; wt += blk) {
        // stage 1: x * 1/sqrt(E[x^2]+eps)  (per-row scalar broadcast over columns)
        const uint32_t norm_cb = do_gamma ? cb_x_normed : cb_out;
        reconfig_data_format(cb_inp, cb_recip_sqrt_var);
        pack_reconfig_data_format(norm_cb);
        mul_bcast_cols_init_short(cb_inp, cb_recip_sqrt_var);
        cb_reserve_back(norm_cb, blk);
        ACQ();
        for (uint32_t wtr = 0; wtr < blk; wtr++) {
            mul_tiles_bcast_cols(cb_inp, cb_recip_sqrt_var, wt + wtr, 0, wtr);
            pack_tile(wtr, norm_cb);
        }
        REL();
        cb_push_back(norm_cb, blk);

        if constexpr (do_gamma) {
            // stage 2: x_normed * gamma  (gamma broadcast over rows)
            const uint32_t gout_cb = do_beta ? cb_gamma_out : cb_out;
            reconfig_data_format(cb_x_normed, cb_gamma);
            pack_reconfig_data_format(gout_cb);
            mul_bcast_rows_init_short(cb_x_normed, cb_gamma);
            cb_wait_front(cb_x_normed, blk);
            cb_reserve_back(gout_cb, blk);
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                mul_tiles_bcast_rows(cb_x_normed, cb_gamma, wtr, wt + wtr, wtr);
                pack_tile(wtr, gout_cb);
            }
            REL();
            cb_push_back(gout_cb, blk);
            cb_pop_front(cb_x_normed, blk);

            if constexpr (do_beta) {
                // stage 3: (x_normed * gamma) + beta  (beta broadcast over rows)
                reconfig_data_format(cb_gamma_out, cb_beta);
                pack_reconfig_data_format(cb_out);
                add_bcast_rows_init_short(cb_gamma_out, cb_beta);
                cb_wait_front(cb_gamma_out, blk);
                cb_reserve_back(cb_out, blk);
                ACQ();
                for (uint32_t wtr = 0; wtr < blk; wtr++) {
                    add_tiles_bcast_rows(cb_gamma_out, cb_beta, wtr, wt + wtr, wtr);
                    pack_tile(wtr, cb_out);
                }
                REL();
                cb_push_back(cb_out, blk);
                cb_pop_front(cb_gamma_out, blk);
            }
        }
    }
    cb_pop_front(cb_recip_sqrt_var, 1);
}

void kernel_main() {
    // Compile-time args (order must match all_gather_rms_norm_program_factory.cpp: compute_ct_args).
    constexpr uint32_t cb_inp = get_compile_time_arg_val(0);
    constexpr uint32_t cb_reduce = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x2 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(3);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(4);
    constexpr uint32_t cb_beta = get_compile_time_arg_val(5);
    constexpr uint32_t cb_var = get_compile_time_arg_val(6);
    constexpr uint32_t cb_recip_sqrt_var = get_compile_time_arg_val(7);
    constexpr uint32_t cb_x_normed = get_compile_time_arg_val(8);
    constexpr uint32_t cb_gamma_out = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out = get_compile_time_arg_val(10);
    constexpr uint32_t cb_local_stats = get_compile_time_arg_val(11);
    constexpr uint32_t cb_gathered_stats = get_compile_time_arg_val(12);
    constexpr uint32_t Wt = get_compile_time_arg_val(13);
    constexpr uint32_t blk = get_compile_time_arg_val(14);
    constexpr uint32_t ring_size = get_compile_time_arg_val(15);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(16);
    constexpr uint32_t do_beta = get_compile_time_arg_val(17);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(18) == 1;
    constexpr uint32_t cb_reduce_one = get_compile_time_arg_val(19);  // SUM scaler (1.0) for the gather-reduce
    constexpr uint32_t gather_chunk = get_compile_time_arg_val(20);   // rows per batched fabric all-gather (ring>1)

    // Runtime arg: number of tile-rows this worker processes.
    const uint32_t NCHt = get_arg_val<uint32_t>(0);

    (void)FLOAT32_DTYPE;
    if constexpr (ring_size > 1) {
        cb_wait_front(cb_reduce_one, 1);  // reader-provided SUM scaler for summing the gathered partials
    }

    binary_op_init_common(cb_inp, cb_inp, cb_var);

    cb_wait_front(cb_reduce, 1);  // reader-provided AVG reduce scalar (1/reduce_factor)
    cb_wait_front(cb_eps, 1);     // reader-provided epsilon
    // gamma / beta are read once by the reader and stay resident for the whole kernel (broadcast over rows).
    if constexpr (do_gamma) {
        cb_wait_front(cb_gamma, Wt);
    }
    if constexpr (do_beta) {
        cb_wait_front(cb_beta, Wt);
    }

    if constexpr (ring_size == 1) {
        // ---- single device: one fused pass per row (input read once, kept resident) ----
        for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
            cb_wait_front(cb_inp, Wt);
            // AVG-reduce x^2 over the full width directly into cb_var = global mean E[x^2].
            reduce_x2(cb_inp, cb_x2, cb_reduce, cb_var, Wt, blk);
            finalize_row<do_gamma, do_beta>(
                cb_var,
                cb_eps,
                cb_recip_sqrt_var,
                cb_inp,
                cb_gamma,
                cb_beta,
                cb_x_normed,
                cb_gamma_out,
                cb_out,
                Wt,
                blk);
            cb_pop_front(cb_inp, Wt);
        }
    } else {
        // ---- multi device: chunked two-pass so the cross-device stats all-gather is batched ----
        // The whole chunk's input (rows * Wt tiles) is read ONCE by the reader and kept RESIDENT in cb_inp
        // across the gather, so pass 2 normalizes it without a DRAM re-stream (halves the activation reads).
        // pass 1: x^2 -> per-device partial (AVG over local width, 1/total_W scaler) into cb_local_stats.
        // [writer batches the fabric all-gather of the chunk's partials -> cb_gathered_stats]
        // pass 2: SUM the ring_size gathered partials -> global E[x^2] -> rsqrt -> normalize.
        for (uint32_t chunk_start = 0; chunk_start < NCHt; chunk_start += gather_chunk) {
            const uint32_t rows = (NCHt - chunk_start) < gather_chunk ? (NCHt - chunk_start) : gather_chunk;

            // Wait for the whole chunk once; pass 1 reads each row by offset (base = r*Wt) WITHOUT popping,
            // so the rows stay resident for pass 2.
            cb_wait_front(cb_inp, rows * Wt);
            for (uint32_t r = 0; r < rows; r++) {
                reduce_x2(cb_inp, cb_x2, cb_reduce, cb_local_stats, Wt, blk, /*base=*/r * Wt);
            }
#ifdef AGRMS_TRACE_LOCAL
            // Print this device's just-computed local partial (cb_local_stats) at the compute's OUTPUT,
            // before the writer copies/mcasts it. Sane here + garbage at the writer => compute->writer
            // handshake race; garbage here => the compute (or its cb_inp input) is already bad.
            if (chunk_start == 0) {
                UNPACK(DPRINT << "AGRMS_LOCAL\n");
                UNPACK(tt::compute::common::print_full_tile(cb_local_stats, 0));
            }
#endif

            // pass 2: normalize the resident rows from the front, popping Wt per row (so finalize_row, which
            // indexes from offset 0, always sees the current row). After the chunk, rows*Wt tiles are popped,
            // matching the single reader push.
            for (uint32_t r = 0; r < rows; r++) {
#ifdef AGRMS_TRACE_LOCAL
                // Format-aware (bf16 tiled) print of the gathered partials the compute is about to SUM:
                // slot[s] is ring peer s's E[x^2] contribution. cb_local_stats was sane at the compute's
                // output; if these gathered slots are garbage IN-MODEL, the fabric gather corrupted them.
                if (chunk_start == 0 && r == 0) {
                    cb_wait_front(cb_gathered_stats, ring_size);
                    for (uint32_t sl = 0; sl < ring_size; sl++) {
                        UNPACK(DPRINT << "AGRMS_GATH slot=" << sl << "\n");
                        UNPACK(tt::compute::common::print_full_tile(cb_gathered_stats, sl));
                    }
                }
#endif
                // SUM the ring_size gathered partials for this row -> global mean E[x^2] in cb_var.
                compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
                    cb_gathered_stats,
                    cb_reduce_one,
                    cb_var,
                    compute_kernel_lib::ReduceInputBlockShape::row(ring_size));
#ifdef AGRMS_TRACE_LOCAL
                if (chunk_start == 0 && r == 0) {
                    UNPACK(DPRINT << "AGRMS_VAR (global E[x^2] = sum of gathered slots)\n");
                    UNPACK(tt::compute::common::print_full_tile(cb_var, 0));
                }
#endif
                finalize_row<do_gamma, do_beta>(
                    cb_var,
                    cb_eps,
                    cb_recip_sqrt_var,
                    cb_inp,
                    cb_gamma,
                    cb_beta,
                    cb_x_normed,
                    cb_gamma_out,
                    cb_out,
                    Wt,
                    blk);
#ifdef AGRMS_TRACE_LOCAL
                if (chunk_start == 0 && r == 0) {
                    UNPACK(DPRINT << "AGRMS_OUT (normalized output row, first tile)\n");
                    UNPACK(tt::compute::common::print_full_tile(cb_out, 0));
                }
#endif
                cb_pop_front(cb_inp, Wt);
            }
        }
    }

    cb_pop_front(cb_eps, 1);
    cb_pop_front(cb_reduce, 1);
    if constexpr (ring_size > 1) {
        cb_pop_front(cb_reduce_one, 1);
    }
    if constexpr (do_gamma) {
        cb_pop_front(cb_gamma, Wt);
    }
    if constexpr (do_beta) {
        cb_pop_front(cb_beta, Wt);
    }
}
