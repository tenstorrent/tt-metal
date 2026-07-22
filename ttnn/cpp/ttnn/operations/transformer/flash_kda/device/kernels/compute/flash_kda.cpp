// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for Flash KDA (Kimi Delta Attention) recurrent state update.
//
// Single-shot per core (no chunk loop — the caller invokes this op once per token,
// carrying S_prev/S_new across calls). Per core, this kernel performs:
//   1. S_tilde = S_prev * g            (decay: g varies per key-dim row, replicated
//                                        across the value columns of that row)
//   2. pred    = k @ S_tilde           ([1,Dk] row-vector contracted against [Dk,Dv])
//   3. err     = v - pred
//   4. delta   = beta * err            (beta is a [1,1] scalar)
//   5. S_new   = S_tilde + (k outer delta)
//   6. out     = q @ S_new             (same contraction pattern as step 2)
//
// Compile-time args: Kt (key-dim tiles, Dk/32), Vt (value-dim tiles, Dv/32)

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"

// CB indices (must match program factory)
constexpr std::uint32_t cb_S_prev = tt::CBIndex::c_0;
constexpr std::uint32_t cb_g = tt::CBIndex::c_1;
constexpr std::uint32_t cb_k = tt::CBIndex::c_2;
constexpr std::uint32_t cb_v = tt::CBIndex::c_3;
constexpr std::uint32_t cb_beta = tt::CBIndex::c_4;
constexpr std::uint32_t cb_q = tt::CBIndex::c_5;

constexpr std::uint32_t cb_k_col = tt::CBIndex::c_6;         // scratch: k, transposed into column layout
constexpr std::uint32_t cb_S_tilde = tt::CBIndex::c_7;       // scratch: S_prev * g
constexpr std::uint32_t cb_pred = tt::CBIndex::c_8;          // scratch: k @ S_tilde
constexpr std::uint32_t cb_err = tt::CBIndex::c_9;           // scratch: v - pred
constexpr std::uint32_t cb_delta = tt::CBIndex::c_10;        // scratch: beta * err
constexpr std::uint32_t cb_delta_bcast = tt::CBIndex::c_11;  // scratch: delta replicated down all 32 rows

constexpr std::uint32_t cb_S_new = tt::CBIndex::c_12;  // output 0: S_tilde + (k outer delta)
constexpr std::uint32_t cb_out = tt::CBIndex::c_13;    // output 1: q @ S_new

constexpr std::uint32_t cb_outer = tt::CBIndex::c_14;  // scratch: k outer delta

void kernel_main() {
    constexpr std::uint32_t Kt = get_compile_time_arg_val(0);
    constexpr std::uint32_t Vt = get_compile_time_arg_val(1);

    constexpr std::uint32_t state_tiles = Kt * Vt;

    // compute_kernel_hw_startup() is the kernel's one-time HW bring-up: it must be the
    // very first Compute-API call, before any wait_front or init (see its own doc
    // comment), so it precedes even the first CB wait below.
    compute_kernel_hw_startup(cb_k, cb_k_col);

    // ------------------------------------------------------------------
    // 0. k_col = transpose(k), one 32x32 tile at a time.
    //    k is given in row layout (one value per column, row 0 only, per Kt tile) —
    //    correct as-is for the two matmuls below, but the outer product needs k in
    //    column layout (one value per row/partition) so it can be the COL-broadcast
    //    operand.
    // ------------------------------------------------------------------
    CircularBuffer(cb_k).wait_front(Kt);
    transpose_init(cb_k);
    CircularBuffer(cb_k_col).reserve_back(Kt);
    for (std::uint32_t kt = 0; kt < Kt; kt++) {
        tile_regs_acquire();
        transpose_tile(cb_k, kt, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_k_col, kt);
        tile_regs_release();
    }
    CircularBuffer(cb_k_col).push_back(Kt);

    // ------------------------------------------------------------------
    // 1. S_tilde = S_prev * g   (COL broadcast: g varies per key-dim row, replicated
    //    across all Vt*32 value columns of that row)
    // ------------------------------------------------------------------
    CircularBuffer(cb_S_prev).wait_front(state_tiles);
    CircularBuffer(cb_g).wait_front(Kt);
    CircularBuffer(cb_S_tilde).reserve_back(state_tiles);
    mul_bcast_cols_init_short(cb_S_prev, cb_g);
    for (std::uint32_t kt = 0; kt < Kt; kt++) {
        for (std::uint32_t vt = 0; vt < Vt; vt++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_S_prev, cb_g, kt * Vt + vt, kt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_S_tilde, kt * Vt + vt);
            tile_regs_release();
        }
    }
    CircularBuffer(cb_S_tilde).push_back(state_tiles);
    CircularBuffer(cb_S_prev).pop_front(state_tiles);
    CircularBuffer(cb_g).pop_front(Kt);

    // ------------------------------------------------------------------
    // 2. pred = k @ S_tilde   ([1,Dk] x [Dk,Dv] -> [1,Dv], contract over Kt)
    // ------------------------------------------------------------------
    CircularBuffer(cb_S_tilde).wait_front(state_tiles);
    CircularBuffer(cb_pred).reserve_back(Vt);
    // Deliberately the deprecated 3-arg mm_init(in0,in1,out) (full HW bring-up on every
    // call), not the newer compute_kernel_hw_startup<SrcOrder::Reverse> + matmul_init
    // split — matches gated_delta_attn.cpp, which mixes matmul with the same bcast/add/sub
    // op types this kernel does and is already known to work with this call.
    mm_init(cb_k, cb_S_tilde, cb_pred);
    for (std::uint32_t vt = 0; vt < Vt; vt++) {
        tile_regs_acquire();
        for (std::uint32_t kt = 0; kt < Kt; kt++) {
            matmul_tiles(cb_k, cb_S_tilde, kt, kt * Vt + vt, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_pred, vt);
        tile_regs_release();
    }
    CircularBuffer(cb_pred).push_back(Vt);
    CircularBuffer(cb_k).pop_front(Kt);

    // ------------------------------------------------------------------
    // 3. err = v - pred
    // ------------------------------------------------------------------
    CircularBuffer(cb_v).wait_front(Vt);
    CircularBuffer(cb_pred).wait_front(Vt);
    CircularBuffer(cb_err).reserve_back(Vt);
    sub_tiles_init(cb_v, cb_pred);
    for (std::uint32_t t = 0; t < Vt; t++) {
        tile_regs_acquire();
        sub_tiles(cb_v, cb_pred, t, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_err, t);
        tile_regs_release();
    }
    CircularBuffer(cb_err).push_back(Vt);
    CircularBuffer(cb_v).pop_front(Vt);
    CircularBuffer(cb_pred).pop_front(Vt);

    // ------------------------------------------------------------------
    // 4. delta = beta * err   (SCALAR broadcast: beta is a single value)
    // ------------------------------------------------------------------
    CircularBuffer(cb_err).wait_front(Vt);
    CircularBuffer(cb_beta).wait_front(1);
    CircularBuffer(cb_delta).reserve_back(Vt);
    mul_tiles_bcast_scalar_init_short(cb_err, cb_beta);
    for (std::uint32_t t = 0; t < Vt; t++) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(cb_err, cb_beta, t, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_delta, t);
        tile_regs_release();
    }
    CircularBuffer(cb_delta).push_back(Vt);
    CircularBuffer(cb_err).pop_front(Vt);
    CircularBuffer(cb_beta).pop_front(1);

    // ------------------------------------------------------------------
    // 5. delta_bcast = replicate delta down all 32 rows (ROW broadcast copy —
    //    single-operand: delta becomes the same for every row/partition, varying
    //    only per column). Needed so the outer product can be formed as a plain
    //    elementwise multiply against the COL-broadcast of k below.
    // ------------------------------------------------------------------
    CircularBuffer(cb_delta).wait_front(Vt);
    CircularBuffer(cb_delta_bcast).reserve_back(Vt);
    unary_bcast_init<BroadcastType::ROW>(cb_delta, cb_delta_bcast);
    for (std::uint32_t t = 0; t < Vt; t++) {
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(cb_delta, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_delta_bcast, t);
        tile_regs_release();
    }
    CircularBuffer(cb_delta_bcast).push_back(Vt);
    CircularBuffer(cb_delta).pop_front(Vt);

    // ------------------------------------------------------------------
    // 6a. outer = k outer delta = delta_bcast[row,col] * k_col[row]  (COL broadcast:
    //     k_col varies per key-dim row, so multiplying it into delta_bcast[row,col]
    //     (= delta[col] for every row) yields k[row]*delta[col], the true outer
    //     product, one Kt tile at a time.
    // ------------------------------------------------------------------
    CircularBuffer(cb_k_col).wait_front(Kt);
    CircularBuffer(cb_outer).reserve_back(state_tiles);
    mul_bcast_cols_init_short(cb_delta_bcast, cb_k_col);
    for (std::uint32_t kt = 0; kt < Kt; kt++) {
        for (std::uint32_t vt = 0; vt < Vt; vt++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_delta_bcast, cb_k_col, vt, kt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_outer, kt * Vt + vt);
            tile_regs_release();
        }
    }
    CircularBuffer(cb_outer).push_back(state_tiles);
    CircularBuffer(cb_delta_bcast).pop_front(Vt);
    CircularBuffer(cb_k_col).pop_front(Kt);

    // ------------------------------------------------------------------
    // 6b. S_new = S_tilde + outer
    // ------------------------------------------------------------------
    CircularBuffer(cb_S_tilde).wait_front(state_tiles);
    CircularBuffer(cb_outer).wait_front(state_tiles);
    CircularBuffer(cb_S_new).reserve_back(state_tiles);
    add_tiles_init(cb_S_tilde, cb_outer);
    for (std::uint32_t t = 0; t < state_tiles; t++) {
        tile_regs_acquire();
        add_tiles(cb_S_tilde, cb_outer, t, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_S_new, t);
        tile_regs_release();
    }
    CircularBuffer(cb_S_new).push_back(state_tiles);
    CircularBuffer(cb_S_tilde).pop_front(state_tiles);
    CircularBuffer(cb_outer).pop_front(state_tiles);

    // ------------------------------------------------------------------
    // 7. out = q @ S_new   (same contraction pattern as step 2)
    // ------------------------------------------------------------------
    CircularBuffer(cb_q).wait_front(Kt);
    CircularBuffer(cb_S_new).wait_front(state_tiles);
    CircularBuffer(cb_out).reserve_back(Vt);
    mm_init(cb_q, cb_S_new, cb_out);
    for (std::uint32_t vt = 0; vt < Vt; vt++) {
        tile_regs_acquire();
        for (std::uint32_t kt = 0; kt < Kt; kt++) {
            matmul_tiles(cb_q, cb_S_new, kt, kt * Vt + vt, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out, vt);
        tile_regs_release();
    }
    CircularBuffer(cb_out).push_back(Vt);
    CircularBuffer(cb_q).pop_front(Kt);
    // cb_S_new is left available (wait_front'd, not popped) for the writer to read.
}
