// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// genify_matmul reader (NCRISC / NoC0) — block operation `load_kblocks` of regime R1.
//
// For every block of this core's region (b-block, m-block, n-block; n fastest) and every K-block,
// reads A[block_m x block_k] and B[block_k x block_n] from interleaved DRAM/L1 through
// TensorAccessor, ONE barrier per K-block covering both operands, one reserve/push per operand.
// Only valid pages are read (rows < tensor_m, cols < tensor_n, K tiles < last_block_k on the last
// K-block) but the NOMINAL page count is always pushed, so the CB rings stay closed and the compute
// kernel runs the nominal block shape (ragged K is excluded in compute by the inner-dim functor).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_act_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_batches = get_compile_time_arg_val(2);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t block_n_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(6);
    constexpr uint32_t last_block_k_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t a_tile_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t b_tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t a_page_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t b_page_bytes = get_compile_time_arg_val(11);
    constexpr auto a_args = TensorAccessorArgs<12>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();

    constexpr uint32_t act_block_tiles = block_m_tiles * block_k_tiles;
    constexpr uint32_t weight_block_tiles = block_k_tiles * block_n_tiles;

    uint32_t arg = 0;
    const uint32_t a_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t b_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t b0 = get_arg_val<uint32_t>(arg++);
    const uint32_t batches_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t row0 = get_arg_val<uint32_t>(arg++);
    const uint32_t rows_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t col0 = get_arg_val<uint32_t>(arg++);
    const uint32_t cols_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t nb_b = get_arg_val<uint32_t>(arg++);
    const uint32_t nb_m = get_arg_val<uint32_t>(arg++);
    const uint32_t nb_n = get_arg_val<uint32_t>(arg++);
    const uint32_t tensor_m_tiles = get_arg_val<uint32_t>(arg++);
    const uint32_t tensor_k_tiles = get_arg_val<uint32_t>(arg++);
    const uint32_t tensor_n_tiles = get_arg_val<uint32_t>(arg++);
    const uint32_t weight_batch_stride = get_arg_val<uint32_t>(arg++);

    const auto a_acc = TensorAccessor(a_args, a_addr, a_page_bytes);
    const auto b_acc = TensorAccessor(b_args, b_addr, b_page_bytes);

    for (uint32_t bb = 0; bb < nb_b; ++bb) {
        const uint32_t blk_b0 = bb * block_batches;
        const uint32_t blk_b_valid =
            (batches_valid - blk_b0) < block_batches ? (batches_valid - blk_b0) : block_batches;
        for (uint32_t mb = 0; mb < nb_m; ++mb) {
            const uint32_t blk_r0 = mb * block_m_tiles;
            const uint32_t blk_rows = (rows_valid - blk_r0) < block_m_tiles ? (rows_valid - blk_r0) : block_m_tiles;
            for (uint32_t nbk = 0; nbk < nb_n; ++nbk) {
                const uint32_t blk_c0 = nbk * block_n_tiles;
                const uint32_t blk_cols = (cols_valid - blk_c0) < block_n_tiles ? (cols_valid - blk_c0) : block_n_tiles;
                // The helper's batch loop consumes block_batches whole K-walks in batch order.
                for (uint32_t bi = 0; bi < block_batches; ++bi) {
                    const bool batch_valid = bi < blk_b_valid;
                    const uint32_t batch = b0 + blk_b0 + bi;
                    const uint32_t a_row_base = batch * tensor_m_tiles + row0 + blk_r0;
                    const uint32_t b_base = batch * weight_batch_stride + col0 + blk_c0;
                    for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                        const uint32_t k_valid = (kb == num_k_blocks - 1) ? last_block_k_tiles : block_k_tiles;
                        const uint32_t k0 = kb * block_k_tiles;

                        cb_reserve_back(cb_act_tiles, act_block_tiles);
                        cb_reserve_back(cb_weight_tiles, weight_block_tiles);
                        if (batch_valid) {
                            // A: CB slot r*block_k + k  <-  page (a_row_base + r) * Kt + k0 + k
                            uint32_t act_l1 = get_write_ptr(cb_act_tiles);
                            for (uint32_t r = 0; r < blk_rows; ++r) {
                                uint32_t page = (a_row_base + r) * tensor_k_tiles + k0;
                                uint32_t l1 = act_l1 + r * block_k_tiles * a_tile_bytes;
                                for (uint32_t k = 0; k < k_valid; ++k) {
                                    noc_async_read(a_acc.get_noc_addr(page + k), l1, a_tile_bytes);
                                    l1 += a_tile_bytes;
                                }
                            }
                            // B: CB slot k*block_n + n  <-  page b_base + (k0 + k) * Nt + n
                            uint32_t weight_l1 = get_write_ptr(cb_weight_tiles);
                            for (uint32_t k = 0; k < k_valid; ++k) {
                                uint32_t page = b_base + (k0 + k) * tensor_n_tiles;
                                uint32_t l1 = weight_l1 + k * block_n_tiles * b_tile_bytes;
                                for (uint32_t n = 0; n < blk_cols; ++n) {
                                    noc_async_read(b_acc.get_noc_addr(page + n), l1, b_tile_bytes);
                                    l1 += b_tile_bytes;
                                }
                            }
                            noc_async_read_barrier();  // one barrier per K-block, both operands
                        }
                        cb_push_back(cb_act_tiles, act_block_tiles);
                        cb_push_back(cb_weight_tiles, weight_block_tiles);
                    }
                }
            }
        }
    }
}
