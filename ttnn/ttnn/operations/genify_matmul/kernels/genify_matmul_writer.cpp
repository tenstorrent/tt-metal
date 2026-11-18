// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// genify_matmul writer (BRISC / NoC1) — block operation `store_block` of regime R1.
//
// Drains the output CB in row groups of sb_h x out_block_n tiles (TileRowMajor, matching the
// matmul_block helper's per-row-group push) and writes the valid tiles to the interleaved output
// (DRAM or L1) through TensorAccessor. Padded rows / columns of ragged edge blocks are skipped.
// One flush per `write_batch` row groups (the CB slot may be reused once the writes have left L1), one
// barrier at the end.
//
// Regime R4 (`out_in_place` = 1): the output CB is backed on this core's own L1 output shard and the
// compute packs straight into it, so the writer only retires the row groups (wait / pop, no NoC).
//
// Regime R4 A rounds (`ds_act` = 1, Refinement 6 DS_ACT_ON_WRITER): this RISC is idle until the block's
// output, so it runs the A broadcast (genify_matmul_ds_act.hpp) on BRISC / NoC1 while the reader streams
// B on NCRISC / NoC0 -- the rounds no longer sit between the B stream's issues and barriers. Per block
// row it produces cb_act_tiles for the K-walk (reserve, round, push), then drains that row's output.
// Cores of the broadcast that are not bank workers (`ds_worker` = 0) send their own rounds and exit.
//
// Regime R1b A stream (`act_mode` = 2, Refinement 9): on a 1-D N split every core needs every A K-block,
// so this RISC produces cb_act_tiles: before each block's output it walks the block's K-blocks, the
// injector (the channel's fixed sender) reading each A K-block from DRAM / L1 into its slot and
// multicasting it to every core's same slot, the others acking and waiting. The reader streams only B.

// Per-stage device zones (Perf 1, PERMANENT -- free with the profiler off; never remove, extend instead):
//   wr_wait_out   cb_wait_front on a row-group batch   (starved: compute has not packed it yet)
//   wr_write      the batch's NoC write issue + flush
//   wr_final_barrier  the closing noc_async_write_barrier
//   wr_tail_wait_reader  (Perf 2) last block: wait for the reader's NoC0 share of a batch to leave L1 before pop
// Perf 2 tail write split (tail_split.hpp): on the core's LAST block the writer posts each output batch to the
// reader (idle once its K-walk is done), writes its own share on NoC1 and pops only after the reader's share
// left L1. PERF FOCUS 1 write stream 60 -> 44 us (17.8 MB from 110 cores; all-NoC1 ~300 GB/s, split ~410).
//   wr_ds_round   one R4 A round (reserve + round + push), per K-block
//   wr_a_inject / wr_a_receive   one R1b A K-walk (injector: reads + multicasts; receiver: acks + waits)
//   wr_a_barrier / wr_a_send     (Refinement 15) the fixed injector's per-K-block read barrier / multicast (incl. acks)
//   wr_a_rotate   one R1b A K-walk under rotating injectors (sends on this core's turns, receives otherwise)
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/ttnn/operations/genify_matmul/kernels/genify_matmul_ds_act.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/ttnn/operations/genify_matmul/kernels/tail_split.hpp"
#include "ttnn/ttnn/operations/genify_matmul/kernels/out_chunks.hpp"
#include "ttnn/ttnn/operations/genify_matmul/kernels/a_segments.hpp"

// R1b bank-local stream (Refinement 15): output column c of a core's region is col0 + c * GM_OUT_STRIDE (the B
// units' bank stride, in output columns); 1 = adjacent columns.
#ifndef GM_OUT_STRIDE
#define GM_OUT_STRIDE 1
#endif
// R1b rotating injectors (Refinement 15, host ACT_STREAM_INJECTORS): the `act` channel's senders take turns, A
// K-block round r is read and multicast by injector r % GM_ACT_INJECTORS. An injector issues the read of its NEXT
// turn (round r + injectors) right after sending round r, so the injectors' A reads overlap each other's multicasts
// and the stream runs at the multicast rate instead of the read + multicast rate of one core. 1 = one fixed injector.
#ifndef GM_ACT_INJECTORS
#define GM_ACT_INJECTORS 1
#endif

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
    // A role of this RISC: 0 none, 1 = R4's A rounds (from resident shards), 2 = R1b's A stream (Refinement 9).
    constexpr uint32_t act_mode = get_compile_time_arg_val(8);
    constexpr bool ds_act = act_mode == 1;      // R4: the A rounds run here
    constexpr bool act_stream = act_mode == 2;  // R1b: the A K-block broadcast runs here
    constexpr bool ds_worker = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t cb_act_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(12);
    constexpr uint32_t k_blocks_per_round = get_compile_time_arg_val(13);
    constexpr uint32_t a_tile_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t act_cb_blocks = get_compile_time_arg_val(15);
    // Write batching (Refinement 8): row groups drained per wait / flush / pop. The output CB holds
    // DEPTH_OUT batches, so compute fills one while the writer's NoC writes of the other are in flight.
    constexpr uint32_t write_batch = get_compile_time_arg_val(16);
    // R1b A stream (Refinement 9): the K-walk's ragged last K-block, A's page size, A residency.
    constexpr uint32_t last_block_k_tiles = get_compile_time_arg_val(17);
    constexpr uint32_t a_page_bytes = get_compile_time_arg_val(18);
    constexpr bool act_resident = get_compile_time_arg_val(19) != 0;
    // R1b injector stall-shadow: A K-blocks the injector keeps in flight (each its own read trid), so the
    // DRAM read of K-block kb+1 overlaps the multicast of kb. 1 = read, barrier, send (no overlap).
    constexpr uint32_t act_ahead = get_compile_time_arg_val(20);
    constexpr auto out_args = TensorAccessorArgs<21>();
    // A's accessor (R1b's injector reads A); every other writer passes the output's args as a stand-in.
    [[maybe_unused]] constexpr auto a_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    constexpr uint32_t row_group_tiles = sb_h * out_block_n_tiles;
    constexpr uint32_t num_row_groups = block_m_tiles / sb_h;
    constexpr uint32_t batch_tiles = write_batch * row_group_tiles;
    constexpr uint32_t batch_rows = write_batch * sb_h;
    static_assert(write_batch >= 1 && num_row_groups % write_batch == 0);

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
    // Output tiles per tile row of one output tensor (N_out, or N_out / chunks for a split output).
    const uint32_t out_n_tiles = get_arg_val<uint32_t>(arg++);
    // R4 A rounds: the A shard's L1 address. R1b: A's buffer address.
    [[maybe_unused]] const uint32_t a_addr = act_mode != 0 ? get_arg_val<uint32_t>(arg++) : 0;
    [[maybe_unused]] const uint32_t tensor_k_tiles = act_stream ? get_arg_val<uint32_t>(arg++) : 0;  // R1b
    // Refinement 14: R1b's A as K segments (a_segments.hpp): the segment table follows (GM_A_SEGMENTS > 1).
    [[maybe_unused]] a_segments::Table a_tab{};
    if constexpr (act_stream) {
        a_tab = a_segments::read_table(a_addr, tensor_k_tiles, arg);
    }
    // Refinement 13 split output: chunk 0 is out_addr, chunks 1.. follow (out_chunks.hpp).
    const out_chunks::Table out_tab = out_chunks::read_table(out_addr, out_n_tiles, arg);

    [[maybe_unused]] const auto a_acc = TensorAccessor(a_args, a_addr, a_page_bytes);

    constexpr auto mc_act = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("act_ct_offset"), get_named_compile_time_arg_val("act_rt_offset")>();
    static_assert(mc_act.active == (act_mode != 0));
    static_assert(!act_stream || block_batches == 1);
    static_assert(act_ahead >= 1 && (!act_stream || act_ahead <= act_cb_blocks));
    constexpr uint32_t act_injectors = GM_ACT_INJECTORS;
    // An injector reserves through its next turn r + injectors while compute may still hold round r (packing its
    // output into a CB this RISC has not drained yet), so the ring must hold injectors + 1 K-blocks.
    static_assert(act_injectors == 1 || (act_stream && !act_resident && act_injectors < act_cb_blocks));
    Noc noc;
    [[maybe_unused]] auto a_sender = mc_act.optional_sender(noc);
    [[maybe_unused]] auto a_receiver = mc_act.optional_receiver(noc);
    [[maybe_unused]] const genify_ds::ActKBlocks<block_m_tiles, block_k_tiles, k_blocks_per_round, a_tile_bytes>
        ds_blocks{a_addr};
    constexpr uint32_t act_block_tiles = block_m_tiles * block_k_tiles;
    [[maybe_unused]] constexpr uint32_t act_block_bytes = act_block_tiles * a_tile_bytes;
    [[maybe_unused]] uint32_t act_seq = 0;  // R1b: A K-blocks pushed so far (the write pointer's ring slot)
    // R1b: A K-block j of the block rows starting at `blk_r0` (`blk_rows` valid) into the ring slot at `l1_slot`.
    [[maybe_unused]] auto read_act_kblock = [&](uint32_t l1_slot, uint32_t blk_r0, uint32_t blk_rows, uint32_t j) {
        const uint32_t k_valid = (j == num_k_blocks - 1) ? last_block_k_tiles : block_k_tiles;
        if constexpr (a_segments::num > 1) {
            const a_segments::Pos at = a_segments::locate(a_tab, j * block_k_tiles);
            for (uint32_t r = 0; r < blk_rows; ++r) {
                a_segments::read_row(
                    a_args,
                    a_tab,
                    a_page_bytes,
                    a_tile_bytes,
                    l1_slot + r * block_k_tiles * a_tile_bytes,
                    row0 + blk_r0 + r,
                    at,
                    k_valid);
            }
        } else {
            for (uint32_t r = 0; r < blk_rows; ++r) {
                const uint32_t page = (row0 + blk_r0 + r) * tensor_k_tiles + j * block_k_tiles;
                uint32_t l1 = l1_slot + r * block_k_tiles * a_tile_bytes;
                for (uint32_t k = 0; k < k_valid; ++k) {
#ifndef GM_ABLATE_A
                    noc_async_read(a_acc.get_noc_addr(page + k), l1, a_tile_bytes);
#endif
                    l1 += a_tile_bytes;
                }
            }
        }
    };
    // Rotating injectors: the ring base (the CB is exactly act_cb_blocks K-blocks) and this core's pending turn.
    [[maybe_unused]] const uint32_t act_ring_base = get_write_ptr(cb_act_tiles);
    [[maybe_unused]] const uint32_t act_rounds = nb_b * nb_m * nb_n * num_k_blocks;
    // Round r's block rows: rounds walk (mb, nbk, kb) with kb fastest (nb_b = block_batches = 1 under R1b).
    [[maybe_unused]] auto round_rows = [&](uint32_t r, uint32_t& blk_r0, uint32_t& blk_rows) {
        blk_r0 = ((r / num_k_blocks) / nb_n % nb_m) * block_m_tiles;
        blk_rows = block_extent(rows_valid, blk_r0, block_m_tiles);
    };
    // Issue this core's read for its turn `r` (r >= act_seq): reserve through r's slot, read under trid 1.
    [[maybe_unused]] auto issue_turn = [&](uint32_t r) {
        cb_reserve_back(cb_act_tiles, (r - act_seq + 1) * act_block_tiles);
        uint32_t blk_r0, blk_rows;
        round_rows(r, blk_r0, blk_rows);
        dataflow_kernel_lib::set_read_trid(noc, 1);
        read_act_kblock(act_ring_base + (r % act_cb_blocks) * act_block_bytes, blk_r0, blk_rows, r % num_k_blocks);
        dataflow_kernel_lib::set_read_trid(noc, 0);
    };
    if constexpr (act_stream && act_injectors > 1) {
        if (a_sender) {
            for (uint32_t r = 0; r < act_injectors && r < act_rounds; ++r) {
                if (mc_act.should_send(r)) {
                    issue_turn(r);  // this injector's first turn
                    break;
                }
            }
        }
    }
    if constexpr (ds_act && !ds_worker) {
        genify_ds::send_only(
            ds_blocks, mc_act, a_sender, get_write_ptr(cb_act_tiles), act_cb_blocks, nb_m, num_k_blocks);
        return;
    }

    // tail_writes_both_nocs: the post / done words (semaphores, zero-initialized).
    [[maybe_unused]] uint32_t tail_seq = 0;
#if TAIL_SPLIT_MODE != 0
    const uint32_t tail_mbox = get_semaphore(TAIL_SEM_MBOX);
    const uint32_t tail_post = get_semaphore(TAIL_SEM_POST);
    const uint32_t tail_done = get_semaphore(TAIL_SEM_DONE);
#endif

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
                    MaybeDeviceZoneScope("wr_ds_round");
                    cb_reserve_back(cb_act_tiles, act_block_tiles);
                    genify_ds::worker_round(
                        ds_blocks, mc_act, a_sender, a_receiver, mb, kb, get_write_ptr(cb_act_tiles));
                    cb_push_back(cb_act_tiles, act_block_tiles);
                }
            }
            for (uint32_t nbk = 0; nbk < nb_n; ++nbk) {
                const uint32_t blk_c0 = nbk * out_block_n_tiles;
                const uint32_t blk_cols = block_extent(cols_valid, blk_c0, out_block_n_tiles);
                const out_chunks::Col blk_col = out_chunks::locate(out_tab, col0 + blk_c0 * GM_OUT_STRIDE);
                if constexpr (act_stream) {
                    // R1b: this block's K-walk of A (once per M-block when A is resident across N-blocks).
                    // Every core walks the same nominal sequence, so the cb_act_tiles write pointers are in
                    // lockstep and the injector multicasts its own freshly read slot to the same address.
                    if constexpr (act_injectors > 1) {
                        MaybeDeviceZoneScope("wr_a_rotate");
                        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                            const uint32_t r = act_seq;
                            if (a_sender && mc_act.should_send(r)) {
                                dataflow_kernel_lib::async_read_barrier_with_trid(noc, 1);  // issued last turn
                                const uint32_t slot = act_ring_base + (r % act_cb_blocks) * act_block_bytes;
#ifdef GM_ABLATE_A_SEND
                                a_sender->send(slot, slot, 32);
#else
                                a_sender->send(slot, slot, act_block_bytes);
#endif
                                cb_push_back(cb_act_tiles, act_block_tiles);
                                ++act_seq;
                                if (r + act_injectors < act_rounds) {
                                    issue_turn(r + act_injectors);  // the next turn's read overlaps others' sends
                                }
                            } else {
                                cb_reserve_back(cb_act_tiles, act_block_tiles);
                                a_receiver->receive(r);
                                cb_push_back(cb_act_tiles, act_block_tiles);
                                ++act_seq;
                            }
                        }
                    } else if (!act_resident || nbk == 0) {
                        if (a_sender) {
                            MaybeDeviceZoneScope("wr_a_inject");
                            // Injector: K-block j lands (j - kb) slots past the write pointer (the CB is
                            // exactly act_cb_blocks slots); its reads carry trid (j % act_ahead) + 1, freed by
                            // retiring j - act_ahead, so retiring kb leaves kb+1 .. kb+ahead-1 on the wire.
                            const uint32_t slot0 = get_write_ptr(cb_act_tiles);
                            const uint32_t ring_base = slot0 - (act_seq % act_cb_blocks) * act_block_bytes;
                            uint32_t issued_from = 0;  // K-block at the write pointer
                            auto issue = [&](uint32_t j) {
                                cb_reserve_back(cb_act_tiles, (j + 1 - issued_from) * act_block_tiles);
                                const uint32_t l1_slot =
                                    ring_base + ((act_seq + j - issued_from) % act_cb_blocks) * act_block_bytes;
                                const uint32_t k_valid = (j == num_k_blocks - 1) ? last_block_k_tiles : block_k_tiles;
                                dataflow_kernel_lib::set_read_trid(noc, j % act_ahead + 1);
                                if constexpr (a_segments::num > 1) {
                                    const a_segments::Pos at = a_segments::locate(a_tab, j * block_k_tiles);
                                    for (uint32_t r = 0; r < blk_rows; ++r) {
                                        a_segments::read_row(
                                            a_args,
                                            a_tab,
                                            a_page_bytes,
                                            a_tile_bytes,
                                            l1_slot + r * block_k_tiles * a_tile_bytes,
                                            row0 + blk_r0 + r,
                                            at,
                                            k_valid);
                                    }
                                } else {
                                    for (uint32_t r = 0; r < blk_rows; ++r) {
                                        const uint32_t page = (row0 + blk_r0 + r) * tensor_k_tiles + j * block_k_tiles;
                                        uint32_t l1 = l1_slot + r * block_k_tiles * a_tile_bytes;
                                        for (uint32_t k = 0; k < k_valid; ++k) {
#ifndef GM_ABLATE_A
                                            noc_async_read(a_acc.get_noc_addr(page + k), l1, a_tile_bytes);
#endif
                                            l1 += a_tile_bytes;
                                        }
                                    }
                                }
                            };
                            for (uint32_t j = 0; j + 1 < act_ahead && j < num_k_blocks; ++j) {
                                issue(j);
                            }
                            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                                if (kb + act_ahead - 1 < num_k_blocks) {
                                    issue(kb + act_ahead - 1);
                                }
                                {
                                    MaybeDeviceZoneScope("wr_a_barrier");
                                    dataflow_kernel_lib::async_read_barrier_with_trid(noc, kb % act_ahead + 1);
                                }
                                const uint32_t slot = get_write_ptr(cb_act_tiles);
                                {
                                    MaybeDeviceZoneScope("wr_a_send");
#ifdef GM_ABLATE_A_SEND
                                    a_sender->send(slot, slot, 32);
#else
                                    a_sender->send(slot, slot, act_block_bytes);
#endif
                                }
                                cb_push_back(cb_act_tiles, act_block_tiles);
                                ++issued_from;
                                ++act_seq;
                            }
                            dataflow_kernel_lib::set_read_trid(noc, 0);  // restore untagged reads
                        } else {
                            MaybeDeviceZoneScope("wr_a_receive");
                            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                                cb_reserve_back(cb_act_tiles, act_block_tiles);
                                a_receiver->receive();
                                cb_push_back(cb_act_tiles, act_block_tiles);
                            }
                        }
                    }
                }
                const bool last_block = bb + 1 == nb_b && mb + 1 == nb_m && nbk + 1 == nb_n;
                for (uint32_t bi = 0; bi < block_batches; ++bi) {
                    const bool batch_valid = bi < blk_b_valid;
                    const uint32_t batch = b0 + blk_b0 + bi;
                    // tail_writes_both_nocs: the LAST batch of the core's LAST block is split with the reader.
                    // Its output exists only once the reader has pushed its final K-block, so the reader is
                    // free to serve it. An earlier batch of a multi-batch block (R1d) must NOT be held for the
                    // reader: the reader is still walking the block's later batches, the held output CB stalls
                    // compute, compute stops freeing the input CBs, and the reader never gets there (Refinement
                    // 12: bf16 PDL H pass, block_batches = 32, deadlock).
                    const bool tail = tail_split::active && !out_in_place && last_block && bi + 1 == block_batches;
                    for (uint32_t g = 0; g < num_row_groups; g += write_batch) {
                        {
                            MaybeDeviceZoneScope("wr_wait_out");
                            cb_wait_front(cb_output_tiles, batch_tiles);
                        }
#if TAIL_SPLIT_MODE != 0
                        if (tail) {
                            ++tail_seq;
                            tail_split::post(tail_mbox, tail_post, get_read_ptr(cb_output_tiles), tail_seq);
                        }
#endif
                        if (!out_in_place && batch_valid) {
                            MaybeDeviceZoneScope("wr_write");
                            const uint32_t l1_base = get_read_ptr(cb_output_tiles);
                            for (uint32_t r = 0; r < batch_rows; ++r) {
                                const uint32_t blk_row = g * sb_h + r;
                                if (blk_row >= blk_rows) {
                                    break;
                                }
                                out_chunks::write_row(
                                    out_args,
                                    out_tab,
                                    o_page_bytes,
                                    o_tile_bytes,
                                    l1_base + r * out_block_n_tiles * o_tile_bytes,
                                    batch * tensor_m_tiles + row0 + blk_r0 + blk_row,
                                    blk_col,
                                    blk_cols,
                                    [&](uint32_t n) { return !tail || !tail_split::reader_owns(n, blk_cols); },
                                    GM_OUT_STRIDE);
                            }
                            noc_async_writes_flushed();  // slot is free once the writes left L1
                        }
#if TAIL_SPLIT_MODE != 0
                        if (tail) {
                            MaybeDeviceZoneScope("wr_tail_wait_reader");
                            tail_split::wait_at_least(tail_done, tail_seq);  // the reader's share left L1
                        }
#endif
                        cb_pop_front(cb_output_tiles, batch_tiles);
                    }
                }
            }
        }
    }
    MaybeDeviceZoneScope("wr_final_barrier");
    noc_async_write_barrier();
}
