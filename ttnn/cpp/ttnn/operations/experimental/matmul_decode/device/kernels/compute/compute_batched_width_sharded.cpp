// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

using std::uint32_t;

// Block-diagonal batched matmul per core. full_in0 is sender-major; matmul_block does not reduce over kt_dim.
//
// With ENABLE_GLOBAL_CB the in1 tiles arrive through a GCB-backed circular buffer instead of a
// globally-allocated one, so this kernel must tell the reader (via the sync CB) that the GCB
// page can be released once every in1 tile of it has been read.
//
// num_k_blocks > 1 means this receiver's slab arrives as that many GCB pages rather than one. The
// slab stacks Bc batches of K rows, so a page holds one or more runs of consecutive rows of a
// single batch, and the traversal below walks those runs rather than whole batches: a page can
// only be read while it is held, and it is popped before the next arrives. A batch that spans a
// page boundary is therefore still half summed when its first page is released, so its running
// sums live in the output CB, where the packer accumulates into them (pack_reconfig_l1_acc).
// num_k_blocks == 1 degenerates to one run per batch, which is the whole-slab traversal this
// kernel has always done.
using namespace ckernel;
void kernel_main() {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t Nc_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t Bc = get_compile_time_arg_val(3);
    constexpr uint32_t inA_K_tiles_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(5);

    // Named so op fusion can remap them; the reader gathers A into cb_full_in0, which is what
    // compute reads as its in0.
    constexpr uint32_t full_in0_cb_id = get_named_compile_time_arg_val("cb_full_in0");
    constexpr uint32_t in1_cb_id = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t out_cb_id = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t sync_cb_id = get_named_compile_time_arg_val("cb_sync");

    constexpr uint32_t full_in0_num_tiles = Bc * M_tiles * K_tiles;
    // One GCB page: page_k_rows consecutive rows of this receiver's [Bc*K, Nc] slab.
    constexpr uint32_t slab_k_rows = Bc * K_tiles;
    constexpr uint32_t page_k_rows = slab_k_rows / num_k_blocks;
    constexpr uint32_t in1_page_tiles = page_k_rows * Nc_tiles;
    constexpr uint32_t out_num_tiles = Bc * M_tiles * Nc_tiles;
    constexpr uint32_t sender_slice_tiles = Bc * M_tiles * inA_K_tiles_per_core;

    constexpr uint32_t out_block_h = M_tiles;
    constexpr uint32_t out_block_w = 1;
    constexpr uint32_t in0_block_w = inA_K_tiles_per_core;

    CircularBuffer full_in0_cb(full_in0_cb_id);
    CircularBuffer in1_cb(in1_cb_id);
    CircularBuffer out_cb(out_cb_id);
#ifdef ENABLE_GLOBAL_CB
    CircularBuffer sync_cb(sync_cb_id);
#endif

    compute_kernel_hw_startup<SrcOrder::Reverse>(full_in0_cb_id, in1_cb_id, out_cb_id);

    full_in0_cb.wait_front(full_in0_num_tiles);

    matmul_block_init(full_in0_cb_id, in1_cb_id, false, out_block_w, out_block_h, in0_block_w);

    out_cb.reserve_back(out_num_tiles);
    for (uint32_t page = 0; page < num_k_blocks; ++page) {
        in1_cb.wait_front(in1_page_tiles);
        const uint32_t page_first_row = page * page_k_rows;
        const uint32_t page_end_row = page_first_row + page_k_rows;
        for (uint32_t row = page_first_row; row < page_end_row;) {
            // The rows this page contributes to one batch: from kt_first up to whichever ends
            // first, the batch or the page.
            const uint32_t bc_i = row / K_tiles;
            const uint32_t kt_first = row - bc_i * K_tiles;
            const uint32_t rows_left_in_batch = K_tiles - kt_first;
            const uint32_t rows_left_in_page = page_end_row - row;
            const uint32_t run_rows = rows_left_in_batch < rows_left_in_page ? rows_left_in_batch : rows_left_in_page;
            if constexpr (num_k_blocks > 1) {
                // A batch that started on an earlier page is half summed; keep adding to it.
                pack_reconfig_l1_acc(kt_first != 0 ? 1 : 0);
            }
            const uint32_t in0_batch_base = (bc_i * M_tiles) * inA_K_tiles_per_core;
            // in1 is indexed within the page currently at the front of the CB.
            const uint32_t in1_run_base = (row - page_first_row) * Nc_tiles;
            for (uint32_t nc = 0; nc < Nc_tiles; ++nc) {
                tile_regs_acquire();
                for (uint32_t i = 0; i < run_rows; ++i) {
                    const uint32_t kt = kt_first + i;
                    const uint32_t sender = kt / inA_K_tiles_per_core;
                    const uint32_t kc_local = kt - sender * inA_K_tiles_per_core;
                    const uint32_t in0_tile = sender * sender_slice_tiles + in0_batch_base + kc_local;
                    const uint32_t in1_tile = in1_run_base + i * Nc_tiles + nc;
                    matmul_block(
                        full_in0_cb_id, in1_cb_id, in0_tile, in1_tile, 0, false, out_block_w, out_block_h, in0_block_w);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t mt = 0; mt < out_block_h; ++mt) {
                    pack_tile<true>(mt, out_cb_id, (bc_i * M_tiles + mt) * Nc_tiles + nc);
                }
                tile_regs_release();
            }
            row += run_rows;
        }
#ifdef ENABLE_GLOBAL_CB
        // This page has been read in full; release the local alias and let the reader ack it.
        in1_cb.pop_front(in1_page_tiles);
        sync_cb.reserve_back(1);
        sync_cb.push_back(1);
#endif
    }
    if constexpr (num_k_blocks > 1) {
        pack_reconfig_l1_acc(0);
    }
    out_cb.push_back(out_num_tiles);
    full_in0_cb.pop_front(full_in0_num_tiles);
}
