// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// genify_matmul writer (BRISC / NoC1) — block operation `store_block` of regime R1.
//
// Drains the output CB in row groups of sb_h x out_block_n tiles (TileRowMajor, matching the
// matmul_block helper's per-row-group push) and writes the valid tiles to the interleaved output
// (DRAM or L1) through TensorAccessor. Padded rows / columns of ragged edge blocks are skipped.
// One flush per row group (the CB slot may be reused once the writes have left L1), one barrier at
// the end.
//
// Regime R4 (`out_in_place` = 1): the output CB is backed on this core's own L1 output shard and the
// compute packs straight into it, so the writer only retires the row groups (wait / pop, no NoC).
//
// Regime R4 A rounds (`ds_act` = 1, Refinement 6 DS_ACT_ON_WRITER): this RISC is idle until the block's
// output, so it runs the A broadcast (genify_matmul_ds_act.hpp) on BRISC / NoC1 while the reader streams
// B on NCRISC / NoC0 -- the rounds no longer sit between the B stream's issues and barriers. Per block
// row it produces cb_act_tiles for the K-walk (reserve, round, push), then drains that row's output.
// Cores of the broadcast that are not bank workers (`ds_worker` = 0) send their own rounds and exit.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/ttnn/operations/genify_matmul/kernels/genify_matmul_ds_act.hpp"

namespace {
// Valid extent of a block starting at `start` inside a region of `valid` tiles (0 when the block lies
// wholly in the padded tail of a ragged edge region: the grid-uniform R1a walk visits such blocks).
FORCE_INLINE uint32_t block_extent(uint32_t valid, uint32_t start, uint32_t nominal) {
    if (valid <= start) {
        return 0;
    }
    return (valid - start) < nominal ? (valid - start) : nominal;
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_batches = get_compile_time_arg_val(1);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t out_block_n_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t sb_h = get_compile_time_arg_val(4);
    constexpr uint32_t o_tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t o_page_bytes = get_compile_time_arg_val(6);
    constexpr bool out_in_place = get_compile_time_arg_val(7) != 0;
    constexpr bool ds_act = get_compile_time_arg_val(8) != 0;  // R4: the A rounds run here
    constexpr bool ds_worker = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t cb_act_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(12);
    constexpr uint32_t k_blocks_per_round = get_compile_time_arg_val(13);
    constexpr uint32_t a_tile_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t act_cb_blocks = get_compile_time_arg_val(15);
    constexpr auto out_args = TensorAccessorArgs<16>();

    constexpr uint32_t row_group_tiles = sb_h * out_block_n_tiles;
    constexpr uint32_t num_row_groups = block_m_tiles / sb_h;

    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
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
    const uint32_t out_n_tiles = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t a_addr = ds_act ? get_arg_val<uint32_t>(arg++) : 0;  // R4 A rounds: A shard

    const auto out_acc = TensorAccessor(out_args, out_addr, o_page_bytes);

    constexpr auto mc_act = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("act_ct_offset"), get_named_compile_time_arg_val("act_rt_offset")>();
    static_assert(mc_act.active == ds_act);
    Noc noc;
    [[maybe_unused]] auto a_sender = mc_act.optional_sender(noc);
    [[maybe_unused]] auto a_receiver = mc_act.optional_receiver(noc);
    [[maybe_unused]] const genify_ds::ActKBlocks<block_m_tiles, block_k_tiles, k_blocks_per_round, a_tile_bytes>
        ds_blocks{a_addr};
    constexpr uint32_t act_block_tiles = block_m_tiles * block_k_tiles;
    if constexpr (ds_act && !ds_worker) {
        genify_ds::send_only(
            ds_blocks, mc_act, a_sender, get_write_ptr(cb_act_tiles), act_cb_blocks, nb_m, num_k_blocks);
        return;
    }

    for (uint32_t bb = 0; bb < nb_b; ++bb) {
        const uint32_t blk_b0 = bb * block_batches;
        const uint32_t blk_b_valid =
            (batches_valid - blk_b0) < block_batches ? (batches_valid - blk_b0) : block_batches;
        for (uint32_t mb = 0; mb < nb_m; ++mb) {
            const uint32_t blk_r0 = mb * block_m_tiles;
            const uint32_t blk_rows = block_extent(rows_valid, blk_r0, block_m_tiles);
            if constexpr (ds_act) {
                // This block row's K-walk of A (R4: nb_b = nb_n = block_batches = 1).
                for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                    cb_reserve_back(cb_act_tiles, act_block_tiles);
                    genify_ds::worker_round(
                        ds_blocks, mc_act, a_sender, a_receiver, mb, kb, get_write_ptr(cb_act_tiles));
                    cb_push_back(cb_act_tiles, act_block_tiles);
                }
            }
            for (uint32_t nbk = 0; nbk < nb_n; ++nbk) {
                const uint32_t blk_c0 = nbk * out_block_n_tiles;
                const uint32_t blk_cols = block_extent(cols_valid, blk_c0, out_block_n_tiles);
                for (uint32_t bi = 0; bi < block_batches; ++bi) {
                    const bool batch_valid = bi < blk_b_valid;
                    const uint32_t batch = b0 + blk_b0 + bi;
                    for (uint32_t g = 0; g < num_row_groups; ++g) {
                        cb_wait_front(cb_output_tiles, row_group_tiles);
                        if (!out_in_place && batch_valid) {
                            const uint32_t l1_base = get_read_ptr(cb_output_tiles);
                            for (uint32_t r = 0; r < sb_h; ++r) {
                                const uint32_t blk_row = g * sb_h + r;
                                if (blk_row >= blk_rows) {
                                    break;
                                }
                                const uint32_t page =
                                    (batch * tensor_m_tiles + row0 + blk_r0 + blk_row) * out_n_tiles + col0 + blk_c0;
                                uint32_t l1 = l1_base + r * out_block_n_tiles * o_tile_bytes;
                                for (uint32_t n = 0; n < blk_cols; ++n) {
                                    noc_async_write(l1, out_acc.get_noc_addr(page + n), o_tile_bytes);
                                    l1 += o_tile_bytes;
                                }
                            }
                            noc_async_writes_flushed();  // slot is free once the writes left L1
                        }
                        cb_pop_front(cb_output_tiles, row_group_tiles);
                    }
                }
            }
        }
    }
    noc_async_write_barrier();
}
