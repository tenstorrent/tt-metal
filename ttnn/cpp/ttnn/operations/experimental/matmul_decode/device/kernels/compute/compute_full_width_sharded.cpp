// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

using std::uint32_t;

// C = A @ B per core. full_in0 is sender-major. matmul_block does not reduce over kt_dim; K is accumulated in the loop.
//
// With ENABLE_GLOBAL_CB the in1 tiles arrive through a GCB-backed circular buffer instead of a
// globally-allocated one, so this kernel must actually wait on them and, when done, tell the
// reader (via the sync CB) that the GCB page can be released.
using namespace ckernel;
void kernel_main() {
    constexpr uint32_t out_block_w = 1;

    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_tiles_per_core = get_compile_time_arg_val(2);
    constexpr uint32_t inA_K_tiles_per_core = get_compile_time_arg_val(3);
    constexpr uint32_t sync_cb_id = get_compile_time_arg_val(4);

    constexpr uint32_t out_block_h = M_tiles;
    constexpr uint32_t in0_block_w = inA_K_tiles_per_core;

    constexpr uint32_t in0_cb_id = tt::CBIndex::c_3;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_2;

    constexpr uint32_t in0_num_tiles = M_tiles * K_tiles;
    constexpr uint32_t in1_num_tiles = K_tiles * N_tiles_per_core;
    constexpr uint32_t num_senders = K_tiles / inA_K_tiles_per_core;
    constexpr uint32_t sender_slice_tiles = M_tiles * inA_K_tiles_per_core;

    CircularBuffer in0_cb(in0_cb_id);
    CircularBuffer out_cb(out_cb_id);
#ifdef ENABLE_GLOBAL_CB
    CircularBuffer in1_cb(in1_cb_id);
    CircularBuffer sync_cb(sync_cb_id);
#endif

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb_id, in1_cb_id, out_cb_id);

    in0_cb.wait_front(in0_num_tiles);
#ifdef ENABLE_GLOBAL_CB
    in1_cb.wait_front(in1_num_tiles);
#endif

    matmul_block_init(in0_cb_id, in1_cb_id, false, out_block_w, out_block_h, in0_block_w);

    out_cb.reserve_back(M_tiles * N_tiles_per_core);
    for (uint32_t bw = 0; bw < N_tiles_per_core; ++bw) {
        tile_regs_acquire();
        for (uint32_t sender = 0; sender < num_senders; ++sender) {
            const uint32_t in0_base = sender * sender_slice_tiles;
            for (uint32_t kc = 0; kc < inA_K_tiles_per_core; ++kc) {
                const uint32_t in0_tile = in0_base + kc;
                const uint32_t k_global = sender * inA_K_tiles_per_core + kc;
                const uint32_t in1_tile = k_global * N_tiles_per_core + bw;
                matmul_block(in0_cb_id, in1_cb_id, in0_tile, in1_tile, 0, false, out_block_w, out_block_h, in0_block_w);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t mt = 0; mt < out_block_h; ++mt) {
            pack_tile<true>(mt, out_cb_id, mt * N_tiles_per_core + bw);
        }
        tile_regs_release();
    }
    out_cb.push_back(M_tiles * N_tiles_per_core);

    in0_cb.pop_front(in0_num_tiles);
#ifdef ENABLE_GLOBAL_CB
    // Every in1 tile has been read; release the local alias and let the reader ack the GCB page.
    in1_cb.pop_front(in1_num_tiles);
    sync_cb.reserve_back(1);
    sync_cb.push_back(1);
#endif
}
