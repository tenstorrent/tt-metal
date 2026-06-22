// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm UNIFIED reader — the single reader for all four paths, gated by two
// compile-time args (Refinement 4):
//   - layout_is_rm : TILE (read tiles) vs ROW_MAJOR (read sticks for tilize)
//   - num_partials : 1 = no cross-core gather (Regime A); K > 1 = Regime B
//                    rotating-sender mcast all-gather of the per-core partial Σx²
// The four combinations are:
//   TILE  + num_partials==1  : Regime A row-parallel (read Wt tiles per row)
//   TILE  + num_partials==K  : Regime B W-split (read Wt_s tiles, then all-gather)
//   RM    + num_partials==1  : row-parallel tilize-wrapped (read sticks per block)
//   RM    + num_partials==K  : RM routed through the SAME mcast all-gather
//
// Gamma is read ONCE into a resident CB (the unified resident-gamma model): TILE
// gamma as Wt_gamma_resident column tiles into cb_gamma (synthetic padding tiles
// zeroed for the RM padded shard); ROW_MAJOR gamma as num_chunks stick-chunks into
// cb_gamma_rm (compute tilizes once). Native non-aligned handling: RM input zeroes
// the W-padding columns of the last real tile (and H-padding sticks).
//
// The all-gather is a K-round rotating-sender exchange with Counter staging and
// EXCLUDE_SRC (src==dst local self-copy) — see Refinement 1 notes / mcast_pipe.hpp.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
FORCE_INLINE void zero_l1(uint32_t addr, uint32_t nbytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    const uint32_t n = nbytes >> 2;
    for (uint32_t i = 0; i < n; ++i) {
        p[i] = 0;
    }
}
}  // namespace

void kernel_main() {
    // ---- compile-time args ----
    constexpr uint32_t cb_input_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_local_sumsq = get_compile_time_arg_val(3);        // mcast handoff
    constexpr uint32_t cb_partials_gathered = get_compile_time_arg_val(4);  // mcast gather dest
    constexpr uint32_t Wt = get_compile_time_arg_val(5);                    // shard real tiles
    constexpr uint32_t Wt_gamma_resident = get_compile_time_arg_val(6);     // resident gamma tiles
    constexpr uint32_t has_gamma = get_compile_time_arg_val(7);
    constexpr uint32_t num_partials = get_compile_time_arg_val(8);  // K (1 = Regime A)
    constexpr uint32_t data_ready_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t consumed_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t gamma_is_rm = get_compile_time_arg_val(11);  // gamma.layout == ROW_MAJOR
    constexpr uint32_t cb_gamma_rm = get_compile_time_arg_val(12);
    constexpr uint32_t reduce_block = get_compile_time_arg_val(13);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t W = get_compile_time_arg_val(15);        // true element count along full W
    constexpr uint32_t in_elem = get_compile_time_arg_val(16);  // RM input element bytes (TILE: unused)
    constexpr uint32_t gamma_elem = get_compile_time_arg_val(17);
    constexpr uint32_t layout_is_rm = get_compile_time_arg_val(18);
    constexpr uint32_t cb_rm_in = get_compile_time_arg_val(19);  // RM input stick dest
    // Refinement 9 (Part A): cross-core all-reduce transport selector (Regime B only).
    //   0 = rotating-sender mcast all-gather (baseline; every one of the K cores mcasts its
    //       partial to the K-1 others — O(K) serialized mcast rounds).
    //   1 = root-relay gather-then-broadcast: rank 0 unicast-reads all K-1 peer partials
    //       (one parallel read phase, gated by a "produced" counter), then issues a SINGLE
    //       mcast of the assembled K-tile block to the group — O(1) transport phases vs the
    //       baseline's O(K) rounds. The COMPUTE kernel is unchanged: every core still ends
    //       with all K partials in cb_partials_gathered and runs the same single-reduce
    //       combine, so only this reader leg differs.
    constexpr uint32_t transport_mode = get_compile_time_arg_val(20);
    constexpr uint32_t produced_sem_id = get_compile_time_arg_val(21);  // mode 1/2: peers->root "produced" counter
    // Refinement 9 (Part D): mode-2 reduce-then-broadcast dest/source CB. Holds the single
    // GLOBAL Σx² tile: on the root the reader cb_wait_fronts it (after compute's combine push)
    // and mcasts it; on a peer the reader receives the broadcast tile into it and pushes it
    // (peer compute's finalize then reads it). Same CB index across all cores → same L1 addr,
    // so the sender's dst addr == every receiver's cb_partial_sumsq addr. Unused for modes 0/1.
    constexpr uint32_t cb_partial_sumsq = get_compile_time_arg_val(22);
    constexpr auto input_args = TensorAccessorArgs<23>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    // ---- runtime args (superset; mcast-only args at [7..] read under constexpr) ----
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t input_page_base = get_arg_val<uint32_t>(2);  // TILE first-unit page base
    const uint32_t gamma_page_base = get_arg_val<uint32_t>(3);  // TILE-B gamma shard base
    const uint32_t start_unit = get_arg_val<uint32_t>(4);       // RM: start_block
    const uint32_t num_units = get_arg_val<uint32_t>(5);        // TILE: rows; RM: blocks
    const uint32_t total_sticks = get_arg_val<uint32_t>(6);     // RM only

    using dataflow_kernel_lib::McastRect;
    using dataflow_kernel_lib::PoolType;
    using dataflow_kernel_lib::ReceiverPipe;
    using dataflow_kernel_lib::ReduceDim;
    using dataflow_kernel_lib::SenderPipe;
    using dataflow_kernel_lib::Staging;

    // SUM scaler = 1.0, col-0 (matmul) fill for SUM + REDUCE_ROW.
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>();

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;
    constexpr uint32_t in_tile_row_bytes = TILE_W * in_elem;
    constexpr uint32_t in_padded_chunk_bytes = reduce_block * in_tile_row_bytes;
    constexpr uint32_t gamma_tile_row_bytes = TILE_W * gamma_elem;
    constexpr uint32_t gamma_padded_chunk_bytes = reduce_block * gamma_tile_row_bytes;
    constexpr uint32_t chunk_cols = reduce_block * TILE_W;

    // ---- gamma read ONCE into a resident CB ----
    if constexpr (has_gamma) {
        if constexpr (gamma_is_rm) {
            // ROW_MAJOR gamma (1,1,1,W): read this core's W-shard columns as
            // row-major stick chunks into cb_gamma_rm; compute tilizes once.
            const uint32_t shard_col0 = gamma_page_base * TILE_W;
            const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr);
            for (uint32_t c = 0; c < num_chunks; ++c) {
                const uint32_t col0 = shard_col0 + c * chunk_cols;
                uint32_t valid_cols = (col0 < W) ? (W - col0) : 0;
                if (valid_cols > chunk_cols) {
                    valid_cols = chunk_cols;
                }
                const uint32_t chunk_row_bytes = valid_cols * gamma_elem;
                const uint32_t byte_off = col0 * gamma_elem;
                cb_reserve_back(cb_gamma_rm, reduce_block);
                uint32_t l1 = get_write_ptr(cb_gamma_rm);
                if (chunk_row_bytes > 0) {
                    const uint32_t zstart = chunk_row_bytes & ~3u;
                    if (zstart < gamma_padded_chunk_bytes) {
                        zero_l1(l1 + zstart, gamma_padded_chunk_bytes - zstart);
                    }
                    noc_async_read(gamma_accessor.get_noc_addr(0, byte_off), l1, chunk_row_bytes);
                } else {
                    zero_l1(l1, gamma_padded_chunk_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(cb_gamma_rm, reduce_block);
            }
        } else {
            // TILE gamma (1,1,1,W) -> column tiles. Read Wt real tiles at gamma_page_base;
            // zero (Wt_gamma_resident - Wt) synthetic padding tiles (RM padded shard).
            const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
            const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, gamma_tile_bytes);
            cb_reserve_back(cb_gamma, Wt_gamma_resident);
            uint32_t l1 = get_write_ptr(cb_gamma);
            for (uint32_t gt = 0; gt < Wt_gamma_resident; ++gt) {
                if (gt < Wt) {
                    noc_async_read_tile(gamma_page_base + gt, gamma_accessor, l1);
                } else {
                    zero_l1(l1, gamma_tile_bytes);
                }
                l1 += gamma_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma, Wt_gamma_resident);
        }
    }

    // ---- input read ----
    {
        DeviceZoneScopedN("RDR-input");
        if constexpr (layout_is_rm) {
            // ROW_MAJOR: read sticks per 32-stick block, chunked along W (with W/H zeroing).
            // For Regime B RM, this core owns a W-column band starting at shard_col0
            // (= input_page_base, repurposed for RM; 0 in Regime A). All column maths is
            // clamped against the GLOBAL W, so padding columns contribute 0 and inv_W in
            // compute carries the true full-row element count.
            const uint32_t shard_col0 = input_page_base;
            const auto input_accessor = TensorAccessor(input_args, input_addr);
            for (uint32_t b = 0; b < num_units; ++b) {
                const uint32_t global_block = start_unit + b;
                const uint32_t block_start_stick = global_block * TILE_H;
                uint32_t rows_this_block = total_sticks - block_start_stick;
                if (rows_this_block > TILE_H) {
                    rows_this_block = TILE_H;
                }
                for (uint32_t c = 0; c < num_chunks; ++c) {
                    const uint32_t col0 = shard_col0 + c * chunk_cols;
                    uint32_t valid_cols = (col0 < W) ? (W - col0) : 0;
                    if (valid_cols > chunk_cols) {
                        valid_cols = chunk_cols;
                    }
                    const uint32_t chunk_row_bytes = valid_cols * in_elem;
                    const uint32_t byte_off = col0 * in_elem;

                    cb_reserve_back(cb_rm_in, reduce_block);
                    uint32_t l1 = get_write_ptr(cb_rm_in);
                    for (uint32_t r = 0; r < TILE_H; ++r) {
                        if (r < rows_this_block && chunk_row_bytes > 0) {
                            const uint32_t zstart = chunk_row_bytes & ~3u;
                            if (zstart < in_padded_chunk_bytes) {
                                zero_l1(l1 + zstart, in_padded_chunk_bytes - zstart);
                            }
                            const uint64_t noc_addr = input_accessor.get_noc_addr(block_start_stick + r, byte_off);
                            noc_async_read(noc_addr, l1, chunk_row_bytes);
                        } else {
                            zero_l1(l1, in_padded_chunk_bytes);
                        }
                        l1 += in_padded_chunk_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_rm_in, reduce_block);
                }
            }
        } else {
            // TILE: read Wt tiles per unit (row) into cb_input_resident.
            const uint32_t tile_bytes = get_tile_size(cb_input_resident);
            const auto input_accessor = TensorAccessor(input_args, input_addr, tile_bytes);
            for (uint32_t u = 0; u < num_units; ++u) {
                const uint32_t page_base = input_page_base + u * Wt;
                {
                    DeviceZoneScopedN("RDR-resv");
                    cb_reserve_back(cb_input_resident, Wt);
                }
                uint32_t l1 = get_write_ptr(cb_input_resident);
                {
                    DeviceZoneScopedN("RDR-noc");
                    for (uint32_t wt = 0; wt < Wt; ++wt) {
                        noc_async_read_tile(page_base + wt, input_accessor, l1);
                        l1 += tile_bytes;
                    }
                    noc_async_read_barrier();
                }
                cb_push_back(cb_input_resident, Wt);
            }
        }
    }  // RDR-input

    // ---- (Regime B) cross-core all-reduce of the K partial Σx² over the group rectangle ----
    if constexpr (num_partials > 1) {
        const uint32_t my_rank = get_arg_val<uint32_t>(7);
        const uint32_t rect_x0 = get_arg_val<uint32_t>(8);
        const uint32_t rect_y0 = get_arg_val<uint32_t>(9);
        const uint32_t rect_x1 = get_arg_val<uint32_t>(10);
        const uint32_t rect_y1 = get_arg_val<uint32_t>(11);
        // sender (peer) virtual coords for rank j at args [12 + 2*j, 12 + 2*j + 1];
        // rank 0 (== root for transport_mode==1) coords are at [12, 13].

        // Wait for compute's fully-accumulated local Σx² (single push).
        {
            DeviceZoneScopedN("RDR-ar-wait");
            cb_wait_front(cb_local_sumsq, 1);
        }
        const uint32_t partial_l1 = get_read_ptr(cb_local_sumsq);

        // Gathered partials carry the INTERMEDIATE format (Float32 when fp32_dest_acc_en),
        // which can differ from the input format — stride / transfer with the partials'
        // own tile size, never the input tile size.
        const uint32_t partial_tile_bytes = get_tile_size(cb_partials_gathered);
        Noc noc;
        const McastRect rect{rect_x0, rect_y0, rect_x1, rect_y1};

        DeviceZoneScopedN("RDR-ar-xport");
        if constexpr (transport_mode == 2) {
            // ----- Reduce-then-broadcast (Refinement 9 Part D) -----
            // Like mode 1, the ROOT gathers all K partials into cb_partials_gathered (same
            // "produced"-gated unicast gather). But instead of broadcasting the K-tile block
            // and having every core reduce it, the ROOT COMPUTE reduces the K gathered partials
            // into cb_partial_sumsq (the single global Σx²) and the ROOT READER then mcasts ONLY
            // that 1 tile. Peers receive that single tile directly into cb_partial_sumsq and
            // their compute SKIPS the combine reduce. cb_partial_sumsq sits at the same CB index
            // (→ same L1 address) on every core, so the sender's dst == every receiver's
            // cb_partial_sumsq base (the §9 same-index-same-address invariant).
            Semaphore<> produced(produced_sem_id);
            const uint32_t root_x = get_arg_val<uint32_t>(12);  // rank-0 (root) virtual coords
            const uint32_t root_y = get_arg_val<uint32_t>(13);

            if (my_rank == 0) {
                // ---- GATHER leg (identical to mode 1): assemble the K-tile block for compute ----
                cb_reserve_back(cb_partials_gathered, num_partials);
                const uint32_t gathered_base = get_write_ptr(cb_partials_gathered);
                noc_async_read(
                    get_noc_addr(my_x[noc_index], my_y[noc_index], partial_l1), gathered_base, partial_tile_bytes);
                produced.wait_min(num_partials - 1);
                for (uint32_t r = 1; r < num_partials; ++r) {
                    const uint32_t px = get_arg_val<uint32_t>(12 + 2 * r);
                    const uint32_t py = get_arg_val<uint32_t>(12 + 2 * r + 1);
                    noc_async_read(
                        get_noc_addr(px, py, partial_l1), gathered_base + r * partial_tile_bytes, partial_tile_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(cb_partials_gathered, num_partials);
                cb_pop_front(cb_local_sumsq, 1);

                // ---- BROADCAST leg: wait for root compute's REDUCED global Σx², mcast 1 tile ----
                // Root compute reduces cb_partials_gathered -> cb_partial_sumsq (the existing
                // combine, gated my_rank==0). Wait for that single push, then mcast ONLY that
                // tile. Do NOT pop cb_partial_sumsq — root compute's FINALIZE (CopyTile) pops it.
                // cb_partial_sumsq is double-buffered (size 2), so a future row's push cannot
                // overwrite this tile's L1 while the mcast send reads it.
                cb_wait_front(cb_partial_sumsq, 1);
                const uint32_t reduced_l1 = get_read_ptr(cb_partial_sumsq);
                DEVICE_PRINT("ROOT send dst_l1={}\n", reduced_l1);
                // The mode-2 broadcast moves a cb_partial_sumsq tile (the global Σx²), so stride
                // with THAT CB's tile size — decoupled from cb_partials_gathered (both carry the
                // intermediate format today, but never assume they match).
                const uint32_t reduced_tile_bytes = get_tile_size(cb_partial_sumsq);
                SenderPipe<
                    num_partials - 1,
                    data_ready_sem_id,
                    consumed_sem_id,
                    Staging::Counter,
                    /*PRE_HANDSHAKE=*/true>
                    sender(noc, rect);
                sender.send(reduced_l1, reduced_l1, reduced_tile_bytes);
            } else {
                // Peer: signal "produced" to root, then receive the SINGLE reduced global tile
                // directly into cb_partial_sumsq. Peer compute SKIPS the combine reduce and its
                // FINALIZE reads this cb_partial_sumsq tile.
                produced.up(noc, root_x, root_y, 1);
                cb_pop_front(cb_local_sumsq, 1);
                cb_reserve_back(cb_partial_sumsq, 1);
                DEVICE_PRINT("PEER rank={} reserve wptr={}\n", my_rank, get_write_ptr(cb_partial_sumsq));
                ReceiverPipe<data_ready_sem_id, consumed_sem_id, Staging::Counter, /*PRE_HANDSHAKE=*/true> receiver(
                    noc);
                receiver.receive(root_x, root_y);
                cb_push_back(cb_partial_sumsq, 1);
            }
        } else if constexpr (transport_mode == 1) {
            // ----- Root-relay gather-then-broadcast (O(1) transport phases) -----
            // Raw noc_async_read is used for the GATHER leg (rank 0 unicast-reads each peer's
            // cb_local_sumsq): mcast_pipe is mcast-oriented and offers no root-reads-many-peers
            // unicast-gather primitive, so this leg stays raw (documented per the §9 rationale).
            // The single BROADCAST leg reuses SenderPipe/ReceiverPipe (mcast_pipe), as the design
            // directs. cb_partials_gathered sits at the same CB index (→ same L1 address) on every
            // core, so the sender's dst_l1 == every receiver's gathered base.
            Semaphore<> produced(produced_sem_id);
            const uint32_t root_x = get_arg_val<uint32_t>(12);  // rank-0 (root) virtual coords
            const uint32_t root_y = get_arg_val<uint32_t>(13);

            if (my_rank == 0) {
                cb_reserve_back(cb_partials_gathered, num_partials);
                const uint32_t gathered_base = get_write_ptr(cb_partials_gathered);
                // self → slot 0 (local copy)
                noc_async_read(
                    get_noc_addr(my_x[noc_index], my_y[noc_index], partial_l1), gathered_base, partial_tile_bytes);
                // Gate on every peer having PRODUCED its local partial into L1 (barrier-before-
                // signal on the peer side: a peer ups `produced` only after its compute pushed
                // cb_local_sumsq, so the data is in L1 before root reads it). §9: counter host-init 0.
                produced.wait_min(num_partials - 1);
                for (uint32_t r = 1; r < num_partials; ++r) {
                    const uint32_t px = get_arg_val<uint32_t>(12 + 2 * r);
                    const uint32_t py = get_arg_val<uint32_t>(12 + 2 * r + 1);
                    noc_async_read(
                        get_noc_addr(px, py, partial_l1), gathered_base + r * partial_tile_bytes, partial_tile_bytes);
                }
                noc_async_read_barrier();
                // ONE mcast of the full K-tile gathered block to the group (EXCLUDE_SRC: root is
                // in the rect with src==dst, so it never mcasts to itself — §9 never-mcast-to-self).
                SenderPipe<
                    num_partials - 1,
                    data_ready_sem_id,
                    consumed_sem_id,
                    Staging::Counter,
                    /*PRE_HANDSHAKE=*/true>
                    sender(noc, rect);
                sender.send(gathered_base, gathered_base, num_partials * partial_tile_bytes);
                cb_push_back(cb_partials_gathered, num_partials);
            } else {
                // Signal root that my partial is produced (in L1), then receive the broadcast.
                produced.up(noc, root_x, root_y, 1);
                cb_reserve_back(cb_partials_gathered, num_partials);
                ReceiverPipe<data_ready_sem_id, consumed_sem_id, Staging::Counter, /*PRE_HANDSHAKE=*/true> receiver(
                    noc);
                receiver.receive(root_x, root_y);
                cb_push_back(cb_partials_gathered, num_partials);
            }
            cb_pop_front(cb_local_sumsq, 1);
        } else {
            // ----- Baseline rotating-sender mcast all-gather (O(K) rounds) -----
            cb_reserve_back(cb_partials_gathered, num_partials);
            const uint32_t gathered_base = get_write_ptr(cb_partials_gathered);
            const uint32_t my_slot = gathered_base + my_rank * partial_tile_bytes;

            // Fill my own slot locally (lets the sender use src == dst -> EXCLUDE_SRC).
            noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], partial_l1), my_slot, partial_tile_bytes);
            noc_async_read_barrier();

            SenderPipe<num_partials - 1, data_ready_sem_id, consumed_sem_id, Staging::Counter, /*PRE_HANDSHAKE=*/true>
                sender(noc, rect);
            ReceiverPipe<data_ready_sem_id, consumed_sem_id, Staging::Counter, /*PRE_HANDSHAKE=*/true> receiver(noc);

            for (uint32_t j = 0; j < num_partials; ++j) {
                if (j == my_rank) {
                    sender.send(my_slot, my_slot, partial_tile_bytes);
                } else {
                    const uint32_t sx = get_arg_val<uint32_t>(12 + 2 * j);
                    const uint32_t sy = get_arg_val<uint32_t>(12 + 2 * j + 1);
                    receiver.receive(sx, sy);
                }
            }

            cb_push_back(cb_partials_gathered, num_partials);
            cb_pop_front(cb_local_sumsq, 1);
        }
    }
}
