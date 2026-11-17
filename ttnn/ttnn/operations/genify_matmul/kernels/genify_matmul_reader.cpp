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
//
// Epilogue operands (Refinement 2):
//   - swiglu: within each N-block, CB column c <- B column n0 + (c < h ? 2c : 2(c - h) + 1),
//     h = block_n / 2, so all gate tiles of the block precede all up tiles (the compute epilogue
//     strides over the two halves).
//   - bias / gate row (`cb_row`): block_n tiles of the (1, N) row, read ONCE per block, pushed right
//     after the first batch's K-walk (so it never stalls the K-block prefetch; compute waits for it
//     before the first epilogue and pops it after the last batch of the block).
//   - residual (addcmul): row groups of sb_h x block_n tiles after each batch's K-walk, in the
//     epilogue's consumption order; depth 2 row groups.

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
    constexpr uint32_t epilogue = get_compile_time_arg_val(12);  // 0 none, 1 bias, 2 bias_relu, 3 swiglu, 4 addcmul
    constexpr uint32_t cb_row = get_compile_time_arg_val(13);
    constexpr uint32_t cb_residual_tiles = get_compile_time_arg_val(14);
    constexpr uint32_t sb_h = get_compile_time_arg_val(15);
    constexpr uint32_t row_tile_bytes = get_compile_time_arg_val(16);
    constexpr uint32_t row_page_bytes = get_compile_time_arg_val(17);
    constexpr uint32_t res_tile_bytes = get_compile_time_arg_val(18);
    constexpr uint32_t res_page_bytes = get_compile_time_arg_val(19);
    constexpr auto a_args = TensorAccessorArgs<20>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto row_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto res_args = TensorAccessorArgs<row_args.next_compile_time_args_offset()>();

    constexpr bool is_swiglu = epilogue == 3;
    constexpr bool has_row = epilogue == 1 || epilogue == 2 || epilogue == 4;
    constexpr bool has_residual = epilogue == 4;
    constexpr uint32_t half_n = block_n_tiles / 2;
    constexpr uint32_t num_row_groups = block_m_tiles / sb_h;
    constexpr uint32_t residual_group_tiles = sb_h * block_n_tiles;

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
    [[maybe_unused]] const uint32_t row_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t res_addr = get_arg_val<uint32_t>(arg++);

    const auto a_acc = TensorAccessor(a_args, a_addr, a_page_bytes);
    const auto b_acc = TensorAccessor(b_args, b_addr, b_page_bytes);
    [[maybe_unused]] const auto row_acc = TensorAccessor(row_args, row_addr, row_page_bytes);
    [[maybe_unused]] const auto res_acc = TensorAccessor(res_args, res_addr, res_page_bytes);

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
                            // B: CB slot k*block_n + c  <-  page b_base + (k0 + k) * Nt + col(c)
                            // (col(c) = c, or the swiglu gate/up permutation).
                            uint32_t weight_l1 = get_write_ptr(cb_weight_tiles);
                            for (uint32_t k = 0; k < k_valid; ++k) {
                                uint32_t page = b_base + (k0 + k) * tensor_n_tiles;
                                uint32_t l1 = weight_l1 + k * block_n_tiles * b_tile_bytes;
                                if constexpr (is_swiglu) {
                                    // blk_cols is even: pair p = (gate 2p, up 2p+1) -> slots p, h + p.
                                    for (uint32_t p = 0; p < blk_cols / 2; ++p) {
                                        noc_async_read(
                                            b_acc.get_noc_addr(page + 2 * p), l1 + p * b_tile_bytes, b_tile_bytes);
                                        noc_async_read(
                                            b_acc.get_noc_addr(page + 2 * p + 1),
                                            l1 + (half_n + p) * b_tile_bytes,
                                            b_tile_bytes);
                                    }
                                } else {
                                    for (uint32_t n = 0; n < blk_cols; ++n) {
                                        noc_async_read(b_acc.get_noc_addr(page + n), l1, b_tile_bytes);
                                        l1 += b_tile_bytes;
                                    }
                                }
                            }
                            noc_async_read_barrier();  // one barrier per K-block, both operands
                        }
                        cb_push_back(cb_act_tiles, act_block_tiles);
                        cb_push_back(cb_weight_tiles, weight_block_tiles);
                    }

                    if constexpr (has_row) {
                        // bias / gate row of this N-block: once per block, after the first K-walk.
                        if (bi == 0) {
                            cb_reserve_back(cb_row, block_n_tiles);
                            uint32_t l1 = get_write_ptr(cb_row);
                            const uint32_t page = col0 + blk_c0;
                            for (uint32_t n = 0; n < blk_cols; ++n) {
                                noc_async_read(row_acc.get_noc_addr(page + n), l1, row_tile_bytes);
                                l1 += row_tile_bytes;
                            }
                            noc_async_read_barrier();
                            cb_push_back(cb_row, block_n_tiles);
                        }
                    }
                    if constexpr (has_residual) {
                        // residual rows of this batch, in row groups of sb_h (epilogue order).
                        for (uint32_t g = 0; g < num_row_groups; ++g) {
                            cb_reserve_back(cb_residual_tiles, residual_group_tiles);
                            if (batch_valid) {
                                const uint32_t res_l1 = get_write_ptr(cb_residual_tiles);
                                for (uint32_t r = 0; r < sb_h; ++r) {
                                    const uint32_t blk_row = g * sb_h + r;
                                    if (blk_row >= blk_rows) {
                                        break;
                                    }
                                    const uint32_t page = (a_row_base + blk_row) * tensor_n_tiles + col0 + blk_c0;
                                    uint32_t l1 = res_l1 + r * block_n_tiles * res_tile_bytes;
                                    for (uint32_t n = 0; n < blk_cols; ++n) {
                                        noc_async_read(res_acc.get_noc_addr(page + n), l1, res_tile_bytes);
                                        l1 += res_tile_bytes;
                                    }
                                }
                                noc_async_read_barrier();
                            }
                            cb_push_back(cb_residual_tiles, residual_group_tiles);
                        }
                    }
                }
            }
        }
    }
}
