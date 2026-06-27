// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// matmul reader (NCRISC). Per output block, per K-block:
//   * activation (in0): first column (X==0) reads its row-block from DRAM and
//     multicasts it ACROSS the row; other columns receive it.
//   * weight (in1): first row (Y==0) reads its column-block from DRAM and
//     multicasts it DOWN the column; other rows receive it.
// in0 phase fully precedes in1 phase each K-block (acyclic: in0 confined to
// rows, in1 to columns). Pipes are constructed per-transfer (proven safe — see
// pipe_rotating.cpp): the ReceiverPipe ctor's INVALID always precedes the
// sender's VALID because the sender's mcast is gated behind the receiver's ack.
//
// BATCHED WEIGHT (Refinement 3 — true batched matmul). The outer `for b` loop
// already exists for batched activation. For a SHARED 2D weight every batch
// re-reads the same (K, N) block (weight_batch_stride == 0). For a BATCHED
// weight (..., K, N) whose leading dims match the activation's, the in1 tile-id
// gains a per-batch `b * weight_batch_stride` (= b*Kt*Nt) offset so batch b
// reads weight matrix b. Nothing else changes — same multicast topology, same
// lock-step ordering, activation read unchanged.
//
// NON-TILE-ALIGNED M / K / N (Refinement 2) — NO sub-tile masking here, by
// design. Mt/Kt/Nt are ceil_div tile counts, so the partial last tile along any
// of M/K/N is a real tile this reader gathers in full. ttnn's TILE_LAYOUT
// representation zero-fills the invalid (out-of-logical-shape) region of that
// partial tile at from_torch time — for fp32, bf16 AND bf8b (the host bf8b
// tilize zeros the pad BEFORE the shared-face-exponent is computed, so no
// block-format exponent corruption). Hence the K dot-product over the padded
// K-region is 0*0=0 automatically; no reader-side zeroing is needed. The
// `gm>=Mt` / `gn>=Nt` skips below are PHANTOM WHOLE tiles (a grid block that
// overruns the tensor), a separate concern from the in-bounds partial tile.
// Verified empirically across all dtypes (changelog Refinement 2, probes 015-016).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

using namespace dataflow_kernel_lib;

constexpr uint32_t CB_IN0_ACT = 0;
constexpr uint32_t CB_IN1_WEIGHT = 1;

void kernel_main() {
    // ---- compile-time args (uniform across the grid) ----
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t batch = get_compile_time_arg_val(3);
    constexpr uint32_t block_M_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t block_N_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(6);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t per_core_M_blocks = get_compile_time_arg_val(8);
    constexpr uint32_t per_core_N_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t IN0_MCAST = get_compile_time_arg_val(10);
    constexpr uint32_t IN1_MCAST = get_compile_time_arg_val(11);
    constexpr uint32_t IN0_DATA_READY = get_compile_time_arg_val(12);
    constexpr uint32_t IN0_CONSUMER_READY = get_compile_time_arg_val(13);
    constexpr uint32_t IN1_DATA_READY = get_compile_time_arg_val(14);
    constexpr uint32_t IN1_CONSUMER_READY = get_compile_time_arg_val(15);
    constexpr uint32_t tileA_bytes = get_compile_time_arg_val(16);
    constexpr uint32_t tileB_bytes = get_compile_time_arg_val(17);
    // Refinement 3 — true batched matmul. Per-batch tile offset for the in1
    // (weight) read: Kt*Nt for a batched weight (..., K, N), 0 for a shared 2D
    // weight (every batch re-reads the same block — Phase-0 behavior).
    constexpr uint32_t weight_batch_stride = get_compile_time_arg_val(18);
    constexpr auto a_args = TensorAccessorArgs<19>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();

    // ---- runtime args (per core) ----
    const uint32_t A_addr = get_arg_val<uint32_t>(0);
    const uint32_t B_addr = get_arg_val<uint32_t>(1);
    const uint32_t grid_row = get_arg_val<uint32_t>(2);  // Y (M split)
    const uint32_t grid_col = get_arg_val<uint32_t>(3);  // X (N split)
    const uint32_t in0_c0 = get_arg_val<uint32_t>(4);
    const uint32_t in0_c1 = get_arg_val<uint32_t>(5);
    const uint32_t in0_c2 = get_arg_val<uint32_t>(6);
    const uint32_t in0_c3 = get_arg_val<uint32_t>(7);
    const uint32_t in1_c0 = get_arg_val<uint32_t>(8);
    const uint32_t in1_c1 = get_arg_val<uint32_t>(9);
    const uint32_t in1_c2 = get_arg_val<uint32_t>(10);
    const uint32_t in1_c3 = get_arg_val<uint32_t>(11);

    const bool in0_sender = (grid_col == 0);
    const bool in1_sender = (grid_row == 0);

    constexpr uint32_t in0_block_tiles = block_M_tiles * in0_block_w;
    constexpr uint32_t in1_block_tiles = in0_block_w * block_N_tiles;
    constexpr uint32_t in0_block_bytes = in0_block_tiles * tileA_bytes;
    constexpr uint32_t in1_block_bytes = in1_block_tiles * tileB_bytes;

    Noc noc;
    CircularBuffer cb_in0(CB_IN0_ACT);
    CircularBuffer cb_in1(CB_IN1_WEIGHT);
    const auto a_acc = TensorAccessor(a_args, A_addr, tileA_bytes);
    const auto b_acc = TensorAccessor(b_args, B_addr, tileB_bytes);

    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t local_mb = 0; local_mb < per_core_M_blocks; local_mb++) {
            const uint32_t global_mb = grid_row * per_core_M_blocks + local_mb;
            const uint32_t mb_base = global_mb * block_M_tiles;

            for (uint32_t local_nb = 0; local_nb < per_core_N_blocks; local_nb++) {
                const uint32_t global_nb = grid_col * per_core_N_blocks + local_nb;
                const uint32_t nb_base = global_nb * block_N_tiles;

                for (uint32_t kb = 0; kb < num_k_blocks; kb++) {
                    const uint32_t kb_base = kb * in0_block_w;

                    // -------- in0 (activation, across row) --------
                    cb_in0.reserve_back(in0_block_tiles);
                    if (in0_sender) {
                        for (uint32_t m = 0; m < block_M_tiles; m++) {
                            const uint32_t gm = mb_base + m;
                            if (gm >= Mt) {
                                continue;  // phantom row: leave stale, output skipped
                            }
                            for (uint32_t k = 0; k < in0_block_w; k++) {
                                const uint32_t gk = kb_base + k;
                                const uint32_t idx = m * in0_block_w + k;
                                const uint32_t tid = b * Mt * Kt + gm * Kt + gk;
                                noc.async_read(
                                    a_acc, cb_in0, tileA_bytes, {.page_id = tid}, {.offset_bytes = idx * tileA_bytes});
                            }
                        }
                        noc.async_read_barrier();
                        if constexpr (IN0_MCAST) {
                            SenderPipe<noc_index, IN0_DATA_READY, true, IN0_CONSUMER_READY> pipe(
                                noc, McastRect<>{in0_c0, in0_c1, in0_c2, in0_c3});
                            const uint32_t wr = cb_in0.get_write_ptr();
                            pipe.send(wr, wr, in0_block_bytes);
                        }
                    } else {
                        ReceiverPipe<IN0_DATA_READY, true, IN0_CONSUMER_READY> pipe(noc);
                        pipe.receive(in0_c0, in0_c1);  // sender_x, sender_y
                    }
                    cb_in0.push_back(in0_block_tiles);

                    // -------- in1 (weight, down column) --------
                    cb_in1.reserve_back(in1_block_tiles);
                    if (in1_sender) {
                        for (uint32_t k = 0; k < in0_block_w; k++) {
                            const uint32_t gk = kb_base + k;
                            for (uint32_t n = 0; n < block_N_tiles; n++) {
                                const uint32_t gn = nb_base + n;
                                if (gn >= Nt) {
                                    continue;  // phantom column
                                }
                                const uint32_t idx = k * block_N_tiles + n;
                                // Refinement 3: b*weight_batch_stride is 0 for a
                                // shared 2D weight, Kt*Nt for a batched weight (so
                                // batch b reads weight matrix b).
                                const uint32_t tid = b * weight_batch_stride + gk * Nt + gn;
                                noc.async_read(
                                    b_acc, cb_in1, tileB_bytes, {.page_id = tid}, {.offset_bytes = idx * tileB_bytes});
                            }
                        }
                        noc.async_read_barrier();
                        if constexpr (IN1_MCAST) {
                            SenderPipe<noc_index, IN1_DATA_READY, true, IN1_CONSUMER_READY> pipe(
                                noc, McastRect<>{in1_c0, in1_c1, in1_c2, in1_c3});
                            const uint32_t wr = cb_in1.get_write_ptr();
                            pipe.send(wr, wr, in1_block_bytes);
                        }
                    } else {
                        ReceiverPipe<IN1_DATA_READY, true, IN1_CONSUMER_READY> pipe(noc);
                        pipe.receive(in1_c0, in1_c1);  // sender_x, sender_y
                    }
                    cb_in1.push_back(in1_block_tiles);
                }
            }
        }
    }
}
