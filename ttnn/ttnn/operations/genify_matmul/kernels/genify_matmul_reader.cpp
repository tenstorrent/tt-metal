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
//
// Regime R1a `interleaved_2d_mcast` (Refinement 3, `operand_mcast` = 1): same region, CBs and walk,
// only the operand SOURCE changes. A K-block depends only on the core's row region, B only on its
// column region, so one injector per line reads it from DRAM and multicasts it into the line's
// cb_act_tiles / cb_weight_tiles (ttnn.Mcast + McastArgs, handshake = the receiver's CB slot is
// reserved). `a_sends` / `b_sends` (CT, per kernel group) mark this core as the A-line / B-line
// injector; every other core of the line receives. All cores of a line walk the same nominal block
// sequence (host passes the grid-uniform nb_m / nb_n), so their CB write pointers stay in lockstep
// and the multicast lands at the sender's own slot address. Per K-block every core delivers A, then
// B: no core can block in B(kb) before every core has passed A(kb), so the two channels cannot
// deadlock. The bias / gate row and the residual stay per-core reads (they are tiny or unshared).
//
// Regime R4 `dram_sharded` (Refinement 4, `dram_sharded` = 1): one worker per DRAM bank owns the block
// rows x (its bank's B column shard) and walks K one A shard per round (`k_blocks_per_round` K-blocks
// per shard). Per K-block it (1) issues ONE coalesced read of the K-block's rows from its bank shard
// (the rows of a bank shard are contiguous) -- before waiting on A, since B does not depend on it --
// then (2) runs the A round on the `act` channel: the core owning A shard `round` multicasts the
// K-block from its resident L1 shard into every worker's reserved cb_act_tiles slot (rotating senders,
// ttnn.Mcast + McastArgs); a worker that owns that shard sends instead of receiving. Cores of the
// broadcast that are not workers (`ds_worker` = 0) only send their own rounds (no CB traffic: they
// address the workers' slot as base + (K-block % depth) * slot bytes) or, inside the receiver
// rectangle, just land the payload and exit. B columns stay in natural (gate, up) order for swiglu.
//
// Regime R3 `block_sharded` (Refinement 5, `block_sharded` = 1): A is L1 block-sharded (rows over grid
// rows, K over grid columns), so a core owns the block rows of its grid row x the output columns of its
// grid column (B column slice read from interleaved DRAM, R1's per-tile reads) and walks K one A shard
// per K-block: K-block kb is the shard of grid column kb of its row. On its own turn (kb == own_kb) the
// core pushes its RESIDENT shard rows into cb_act_shard (backed on A's shard: zero-copy, no NoC read);
// compute selects that CB for the block (In0SourceFn). Peer K-slices land in cb_act_tiles:
//   bs_mcast = 0 (R3 stepping stone): unicast reads of the peer's shard tiles through A's (sharded)
//                TensorAccessor; nothing is pushed to cb_act_tiles on the own turn.
//   bs_mcast = 1 (R3a): a rotating-sender row multicast on the `act` channel -- in round kb the core of
//                grid column kb sends its shard rows from L1 to every row peer's reserved cb_act_tiles
//                slot. To keep the row's cb_act_tiles write pointers in lockstep (the multicast lands at
//                the sender's own slot address) the sender also reserves / pushes its slot on its own
//                turn (it receives the loopback copy); compute retires that slot unread.
// B's K-block reads are issued before the A round (they do not depend on it); one barrier per K-block.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

namespace {

// Valid extent of a block starting at `start` inside a region of `valid` tiles (0 when the block lies
// wholly in the padded tail of a ragged edge region: the grid-uniform R1a walk visits such blocks).
FORCE_INLINE uint32_t block_extent(uint32_t valid, uint32_t start, uint32_t nominal) {
    if (valid <= start) {
        return 0;
    }
    return (valid - start) < nominal ? (valid - start) : nominal;
}

// One line pipe per operand: the sender face on the injector, the receiver face elsewhere, nothing
// when the channel is absent or not a line channel (regime R1: every core reads for itself; R4: the
// act channel is the rotating A broadcast, built separately).
template <bool SENDS, bool LINE, typename Channel>
FORCE_INLINE auto make_line_pipe(const Channel& channel, const Noc& noc) {
    if constexpr (!LINE || !Channel::active) {
        return 0;
    } else if constexpr (SENDS) {
        return channel.sender(noc);
    } else {
        return channel.receiver(noc);
    }
}

// Deliver one K-block over the line: the injector multicasts its freshly read slot to the same
// address on every receiver; a receiver acks its reserved slot and waits for the data.
template <bool ACTIVE, bool SENDS, typename Pipe>
FORCE_INLINE void line_deliver(Pipe& pipe, uint32_t l1, uint32_t bytes) {
    if constexpr (ACTIVE) {
        if constexpr (SENDS) {
            pipe.send(l1, l1, bytes);
        } else {
            pipe.receive();
        }
    }
}

}  // namespace

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
    constexpr bool operand_mcast = get_compile_time_arg_val(20) != 0;  // regime R1a
    constexpr bool a_sends = get_compile_time_arg_val(21) != 0;        // A-line injector (R1a)
    constexpr bool b_sends = get_compile_time_arg_val(22) != 0;        // B-line injector (R1a)
    constexpr bool dram_sharded = get_compile_time_arg_val(23) != 0;   // regime R4
    constexpr bool ds_worker = get_compile_time_arg_val(24) != 0;      // R4 bank worker (else sender / lander)
    constexpr uint32_t k_blocks_per_round = get_compile_time_arg_val(25);  // R4: K-blocks per A shard
    constexpr uint32_t act_cb_blocks = get_compile_time_arg_val(26);       // cb_act_tiles depth in K-blocks
    constexpr bool block_sharded = get_compile_time_arg_val(27) != 0;      // regime R3
    constexpr bool bs_mcast = get_compile_time_arg_val(28) != 0;           // R3a: rotating row multicast
    constexpr uint32_t cb_act_shard = get_compile_time_arg_val(29);        // R3: A's resident shard (zero-copy)
    constexpr auto a_args = TensorAccessorArgs<30>();
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
    // R1: every core reads both operands. R1a: only the line injectors read from DRAM.
    constexpr bool reads_act = !operand_mcast || a_sends;
    constexpr bool reads_weight = !operand_mcast || b_sends;

    // A rides the A-line channel, B the B-line channel (absent under R1).
    constexpr auto mc_act = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("act_ct_offset"), get_named_compile_time_arg_val("act_rt_offset")>();
    constexpr auto mc_weight = dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("weight_ct_offset"),
        get_named_compile_time_arg_val("weight_rt_offset")>();
    static_assert(mc_act.active == (operand_mcast || dram_sharded || bs_mcast) && mc_weight.active == operand_mcast);
    static_assert(int(operand_mcast) + int(dram_sharded) + int(block_sharded) <= 1);
    static_assert(!bs_mcast || block_sharded);

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
    [[maybe_unused]] const uint32_t own_kb = get_arg_val<uint32_t>(arg++);  // R3: this core's K shard (grid column)

    const auto a_acc = TensorAccessor(a_args, a_addr, a_page_bytes);
    const auto b_acc = TensorAccessor(b_args, b_addr, b_page_bytes);
    [[maybe_unused]] const auto row_acc = TensorAccessor(row_args, row_addr, row_page_bytes);
    [[maybe_unused]] const auto res_acc = TensorAccessor(res_args, res_addr, res_page_bytes);

    Noc noc;
    [[maybe_unused]] auto act_pipe = make_line_pipe<a_sends, operand_mcast>(mc_act, noc);
    [[maybe_unused]] auto weight_pipe = make_line_pipe<b_sends, operand_mcast>(mc_weight, noc);

    // R4 A broadcast faces (std::nullopt where this core has no such role) and the A K-block source:
    // K-block (row block mb, K-block kb) of the sender's row-major shard (rows x a_shard_k_tiles) starts
    // at row mb*block_m, column (kb % k_blocks_per_round) * block_k -- one contiguous run, because the
    // host splits a shard (k_blocks_per_round > 1) only when block_m == 1.
    [[maybe_unused]] auto a_sender = mc_act.optional_sender(noc);
    [[maybe_unused]] auto a_receiver = mc_act.optional_receiver(noc);
    constexpr uint32_t a_shard_k_tiles = block_k_tiles * k_blocks_per_round;
    constexpr uint32_t act_block_bytes = act_block_tiles * a_tile_bytes;
    [[maybe_unused]] auto a_shard_src = [&](uint32_t mb, uint32_t kb) {
        return a_addr +
               (mb * block_m_tiles * a_shard_k_tiles + (kb % k_blocks_per_round) * block_k_tiles) * a_tile_bytes;
    };
    if constexpr (dram_sharded && !ds_worker) {
        // A shard core that is not a bank worker: send this core's rounds into the workers' slots. The
        // workers' cb_act_tiles write pointer visits slot (K-block % depth) in lockstep from its base.
        if (a_sender) {
            const uint32_t act_base = get_write_ptr(cb_act_tiles);
            uint32_t slot = 0;
            for (uint32_t mb = 0; mb < nb_m; ++mb) {
                for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                    if (mc_act.should_send(kb / k_blocks_per_round)) {
                        a_sender->send(a_shard_src(mb, kb), act_base + slot * act_block_bytes, act_block_bytes);
                    }
                    slot = (slot + 1 == act_cb_blocks) ? 0 : slot + 1;
                }
            }
        }
        return;
    }

    // A K-block: CB slot r*block_k + k  <-  page (a_row_base + r) * Kt + k0 + k (interleaved A under R1;
    // a peer's shard tiles under the R3 unicast stepping stone -- the accessor resolves the owning core).
    auto read_act = [&](uint32_t act_l1, uint32_t a_row_base, uint32_t k0, uint32_t blk_rows, uint32_t k_valid) {
        for (uint32_t r = 0; r < blk_rows; ++r) {
            uint32_t page = (a_row_base + r) * tensor_k_tiles + k0;
            uint32_t l1 = act_l1 + r * block_k_tiles * a_tile_bytes;
            for (uint32_t k = 0; k < k_valid; ++k) {
                noc_async_read(a_acc.get_noc_addr(page + k), l1, a_tile_bytes);
                l1 += a_tile_bytes;
            }
        }
    };
    // B K-block: CB slot k*block_n + c  <-  page b_base + (k0 + k) * Nt + col(c)
    // (col(c) = c, or the swiglu gate/up permutation).
    auto read_weight = [&](uint32_t weight_l1, uint32_t b_base, uint32_t k0, uint32_t k_valid, uint32_t blk_cols) {
        for (uint32_t k = 0; k < k_valid; ++k) {
            uint32_t page = b_base + (k0 + k) * tensor_n_tiles;
            uint32_t l1 = weight_l1 + k * block_n_tiles * b_tile_bytes;
            if constexpr (is_swiglu) {
                // blk_cols is even: pair p = (gate 2p, up 2p+1) -> slots p, h + p.
                for (uint32_t p = 0; p < blk_cols / 2; ++p) {
                    noc_async_read(b_acc.get_noc_addr(page + 2 * p), l1 + p * b_tile_bytes, b_tile_bytes);
                    noc_async_read(
                        b_acc.get_noc_addr(page + 2 * p + 1), l1 + (half_n + p) * b_tile_bytes, b_tile_bytes);
                }
            } else {
                for (uint32_t n = 0; n < blk_cols; ++n) {
                    noc_async_read(b_acc.get_noc_addr(page + n), l1, b_tile_bytes);
                    l1 += b_tile_bytes;
                }
            }
        }
    };

    for (uint32_t bb = 0; bb < nb_b; ++bb) {
        const uint32_t blk_b0 = bb * block_batches;
        const uint32_t blk_b_valid =
            (batches_valid - blk_b0) < block_batches ? (batches_valid - blk_b0) : block_batches;
        for (uint32_t mb = 0; mb < nb_m; ++mb) {
            const uint32_t blk_r0 = mb * block_m_tiles;
            const uint32_t blk_rows = block_extent(rows_valid, blk_r0, block_m_tiles);
            for (uint32_t nbk = 0; nbk < nb_n; ++nbk) {
                const uint32_t blk_c0 = nbk * block_n_tiles;
                const uint32_t blk_cols = block_extent(cols_valid, blk_c0, block_n_tiles);
                // The helper's batch loop consumes block_batches whole K-walks in batch order.
                for (uint32_t bi = 0; bi < block_batches; ++bi) {
                    const bool batch_valid = bi < blk_b_valid;
                    const uint32_t batch = b0 + blk_b0 + bi;
                    const uint32_t a_row_base = batch * tensor_m_tiles + row0 + blk_r0;
                    const uint32_t b_base = batch * weight_batch_stride + col0 + blk_c0;
                    for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                        const uint32_t k_valid = (kb == num_k_blocks - 1) ? last_block_k_tiles : block_k_tiles;
                        const uint32_t k0 = kb * block_k_tiles;

                        if constexpr (dram_sharded) {
                            // B (stall-shadow: issued before the A round it does not depend on): rows
                            // [k0, k0 + k_valid) of this bank's column shard are one contiguous run.
                            cb_reserve_back(cb_weight_tiles, weight_block_tiles);
                            noc_async_read(
                                b_acc.get_noc_addr(k0 * tensor_n_tiles + col0 + blk_c0),
                                get_write_ptr(cb_weight_tiles),
                                k_valid * block_n_tiles * b_tile_bytes);
                            // A round: the owner of A shard `round` multicasts this K-block to every
                            // worker's reserved slot (the receiver acks the slot, then waits the data).
                            cb_reserve_back(cb_act_tiles, act_block_tiles);
                            const uint32_t round = kb / k_blocks_per_round;
                            if (mc_act.should_send(round)) {
                                a_sender->send(a_shard_src(mb, kb), get_write_ptr(cb_act_tiles), act_block_bytes);
                            } else {
                                a_receiver->receive(round);
                            }
                            noc_async_read_barrier();
                            cb_push_back(cb_act_tiles, act_block_tiles);
                            cb_push_back(cb_weight_tiles, weight_block_tiles);
                        } else if constexpr (block_sharded) {
                            cb_reserve_back(cb_weight_tiles, weight_block_tiles);
                            read_weight(get_write_ptr(cb_weight_tiles), b_base, k0, k_valid, blk_cols);
                            const bool own = kb == own_kb;
                            if (own) {
                                // The block rows of this core's own shard are resident: hand them over.
                                cb_reserve_back(cb_act_shard, act_block_tiles);
                            }
                            if constexpr (bs_mcast) {
                                // Every core reserves round kb's slot (the sender too: lockstep pointers).
                                cb_reserve_back(cb_act_tiles, act_block_tiles);
                                if (own) {
                                    a_sender->send(
                                        get_write_ptr(cb_act_shard), get_write_ptr(cb_act_tiles), act_block_bytes);
                                } else {
                                    a_receiver->receive(kb);
                                }
                            } else {
                                if (!own) {
                                    cb_reserve_back(cb_act_tiles, act_block_tiles);
                                    read_act(get_write_ptr(cb_act_tiles), a_row_base, k0, blk_rows, k_valid);
                                }
                            }
                            noc_async_read_barrier();  // one barrier per K-block, both operands
                            if (own) {
                                cb_push_back(cb_act_shard, act_block_tiles);
                            }
                            if (bs_mcast || !own) {
                                cb_push_back(cb_act_tiles, act_block_tiles);
                            }
                            cb_push_back(cb_weight_tiles, weight_block_tiles);
                        } else {
                            cb_reserve_back(cb_act_tiles, act_block_tiles);
                            cb_reserve_back(cb_weight_tiles, weight_block_tiles);
                            const uint32_t act_l1 = get_write_ptr(cb_act_tiles);
                            const uint32_t weight_l1 = get_write_ptr(cb_weight_tiles);
                            if (batch_valid && reads_act) {
                                read_act(act_l1, a_row_base, k0, blk_rows, k_valid);
                            }
                            if (batch_valid && reads_weight) {
                                read_weight(weight_l1, b_base, k0, k_valid, blk_cols);
                            }
                            if constexpr (reads_act || reads_weight) {
                                noc_async_read_barrier();  // one barrier per K-block, both operands
                            }
                            // R1a: A over its line, then B over its line (fixed order on every core).
                            line_deliver<operand_mcast, a_sends>(act_pipe, act_l1, act_block_tiles * a_tile_bytes);
                            line_deliver<operand_mcast, b_sends>(
                                weight_pipe, weight_l1, weight_block_tiles * b_tile_bytes);
                            cb_push_back(cb_act_tiles, act_block_tiles);
                            cb_push_back(cb_weight_tiles, weight_block_tiles);
                        }
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
