// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per (head, seq_idx), this kernel performs:
//   1) projected = state * g                  (broadcast-scalar multiply)
//   2) new_state = factor @ projected + bktv  (matmul fused with elementwise add)
//   3) state    <- new_state                  (recurrence: feed back for next seq_idx)
//
// Tile shapes (all in 32x32 tiles):
//   - state_cb / current_state_cb / projected_cb : [Kt x Vt]
//   - factor_cb                                  : [Kt x Kt]
//   - bktv_cb                                    : [Kt x Vt]
//   - output_cb                                  : [Kt x Vt]
//
// The bktv add is fused into the matmul by pre-loading bktv into DST and letting
// matmul_tiles accumulate factor @ projected on top of it (DST += A*B). This
// keeps output_cb the sole destination for the matmul output (no double-pack).
//
// State recurrence is implemented by the writer kernel: after writing each
// non-final seq_idx's output_cb tiles to DRAM, the writer also issues a local
// L1-to-L1 NOC copy from output_cb into current_state_cb and pushes that CB.
// This compute kernel then reads state for the next seq_idx from current_state_cb
// while the very first seq_idx of each head still reads from the reader-pushed
// initial state in state_cb.
//
// CB contract:
//   - state_cb         : reader pushes ``state_num_tiles`` (Kt*Vt) once per head.
//                        Consumed by compute on seq_idx == 0 only.
//   - current_state_cb : writer pushes ``state_num_tiles`` after each non-final
//                        seq_idx. Consumed by compute on seq_idx > 0.
//   - g_cb             : producer pushes 1 tile per (head, seq_idx).
//   - factor_cb        : producer pushes ``factor_num_tiles`` (Kt*Kt) per (head, seq_idx).
//   - bktv_cb          : producer pushes ``state_num_tiles`` (Kt*Vt) per (head, seq_idx).
//   - projected_cb     : compute produces and immediately consumes ``state_num_tiles``
//                        per (head, seq_idx) as the matmul B operand.
//   - output_cb        : downstream writer frees ``state_num_tiles`` per (head, seq_idx).

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t state_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t g_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t factor_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t bktv_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t projected_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t output_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t current_state_cb_index = get_compile_time_arg_val(6);
    constexpr uint32_t seq_len = get_compile_time_arg_val(7);
    constexpr uint32_t state_num_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t factor_num_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t dim_k = get_compile_time_arg_val(10);
    constexpr uint32_t dim_v = get_compile_time_arg_val(11);

    // Per-core number of heads assigned to this core. Runtime arg so that a
    // single compiled binary can be shared across cores that get different
    // amounts of work (head_offset is unused by compute; it only matters for
    // dataflow DRAM addressing).
    const uint32_t num_heads = get_arg_val<uint32_t>(0);

    constexpr uint32_t tile_hw = 32;
    constexpr uint32_t Kt = dim_k / tile_hw;
    constexpr uint32_t Vt = dim_v / tile_hw;

    constexpr uint32_t onetile = 1;

    DEVICE_PRINT_UNPACK("State CB Index: {}\n", state_cb_index);
    DEVICE_PRINT_UNPACK("G CB Index: {}\n", g_cb_index);
    DEVICE_PRINT_UNPACK("Factor CB Index: {}\n", factor_cb_index);
    DEVICE_PRINT_UNPACK("Projected CB Index: {}\n", projected_cb_index);
    DEVICE_PRINT_UNPACK("Output CB Index: {}\n", output_cb_index);
    DEVICE_PRINT_UNPACK("Num Heads: {}\n", num_heads);
    DEVICE_PRINT_UNPACK("Seq Len: {}\n", seq_len);
    DEVICE_PRINT_UNPACK("State Num Tiles: {}\n", state_num_tiles);
    DEVICE_PRINT_UNPACK("Factor Num Tiles: {}\n", factor_num_tiles);
    DEVICE_PRINT_UNPACK("Kt: {}\n", Kt);
    DEVICE_PRINT_UNPACK("Vt: {}\n", Vt);

    binary_op_init_common(state_cb_index, g_cb_index, output_cb_index);
    mul_tiles_bcast_scalar_init_short(state_cb_index, g_cb_index);
    mm_block_init(factor_cb_index, projected_cb_index, output_cb_index, false, Vt, Kt, Kt);

    for (uint32_t head = 0; head < num_heads; ++head) {
        // Initial state for this head was pushed to state_cb by the reader. Subsequent
        // seq_idx iterations pull state from current_state_cb (filled by the writer).
        cb_wait_front(state_cb_index, state_num_tiles);

        for (uint32_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
            // Pick the state source: reader-pushed initial on the first seq_idx of the
            // head, writer-pushed recurrent state on every subsequent seq_idx.
            const uint32_t state_src_cb = (seq_idx == 0) ? state_cb_index : current_state_cb_index;

            if (seq_idx > 0) {
                cb_wait_front(current_state_cb_index, state_num_tiles);
            }

            // ---- Step 1: projected_cb = state * g (broadcast scalar) ----
            cb_wait_front(g_cb_index, onetile);
            cb_reserve_back(projected_cb_index, state_num_tiles);

            mul_tiles_bcast_scalar_init_short(state_src_cb, g_cb_index);
            for (uint32_t i = 0; i < state_num_tiles; ++i) {
                tile_regs_acquire();
                mul_tiles_bcast_scalar(state_src_cb, g_cb_index, i, 0, 0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile(0, projected_cb_index);
                tile_regs_release();
            }

            cb_push_back(projected_cb_index, state_num_tiles);
            cb_pop_front(g_cb_index, onetile);

            // The state source for this seq_idx has been fully consumed by step 1.
            // Pop it so the producer can refill the slot for the next iteration:
            //   - seq_idx == 0  : pops state_cb (one-shot, reader filled it once per head)
            //   - seq_idx > 0   : pops current_state_cb (writer fills it per seq_idx)
            cb_pop_front(state_src_cb, state_num_tiles);

            // ---- Step 2: output_cb = factor @ projected + bktv (in place in output_cb) ----
            // factor : [Kt x Kt], projected : [Kt x Vt], bktv/output : [Kt x Vt].
            // The add is fused into the matmul by pre-loading the bktv tile into DST
            // and letting matmul_tiles accumulate factor @ projected on top of it.
            cb_wait_front(factor_cb_index, factor_num_tiles);
            cb_wait_front(projected_cb_index, state_num_tiles);
            cb_wait_front(bktv_cb_index, state_num_tiles);
            cb_reserve_back(output_cb_index, state_num_tiles);

            for (uint32_t m = 0; m < Kt; ++m) {
                for (uint32_t n = 0; n < Vt; ++n) {
                    tile_regs_acquire();

                    // Seed DST[0] with bktv[m, n] so the subsequent matmul accumulation
                    // produces DST[0] = bktv + sum_k(factor[m,k] * projected[k,n]).
                    copy_tile_to_dst_init_short(bktv_cb_index);
                    copy_tile(bktv_cb_index, m * Vt + n, 0);

                    // Switch unpacker/math back to matmul mode and accumulate Kt partial
                    // products on top of the seeded bktv tile (matmul_tiles does DST += A*B).
                    mm_init_short(factor_cb_index, projected_cb_index);
                    for (uint32_t k = 0; k < Kt; ++k) {
                        matmul_tiles(factor_cb_index, projected_cb_index, m * Kt + k, k * Vt + n, 0);
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile(0, output_cb_index);
                    tile_regs_release();
                }
            }

            cb_push_back(output_cb_index, state_num_tiles);
            cb_pop_front(factor_cb_index, factor_num_tiles);
            cb_pop_front(projected_cb_index, state_num_tiles);
            cb_pop_front(bktv_cb_index, state_num_tiles);
        }
    }
}
