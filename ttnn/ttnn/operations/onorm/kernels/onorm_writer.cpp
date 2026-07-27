// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// onorm writer (BRISC / NoC1).  TWO jobs, both writes, both therefore on NoC1:
//
//   1. the CROSS-CORE RE-TILE EXCHANGE (Refinement 2, `group_cores > 1` only) —
//      scatter this core's untilized row-major token rows into the `group_cores`
//      cores that own the flat output columns, and publish this core's own
//      re-tile stripe once every member's chunks have landed;
//   2. drain finished flat token-major output tiles from cb_out_tiles to their
//      interleaved DRAM pages, in `dm_block_tiles`-sized groups with ONE
//      noc_async_write_barrier per group (the writer half of the DM_BLOCK_TILES
//      dataflow lever the reader also reads).
//
// The writer deliberately performs NO reads (it does not fetch `gate`, even
// though that would balance per-core byte counts): reads issued on NoC1 measured
// ~4.8x slower than on NoC0.  See op_design.md §6.
//
// ---------------------------------------------------------------------------
// THE EXCHANGE, and why it is shaped like this
// ---------------------------------------------------------------------------
// A flat output tile spans TOKENS_PER_BLOCK tokens, so the re-tile is the one
// dependent axis of the op and 32 tokens is the atomic work unit for a per-block
// split.  This exchange splits that unit anyway, on TWO axes at once:
//
//   * compute normalizes only THIS core's `tokens_per_core` tokens and untilizes
//     them into cb_rm_local — one contiguous `local_row_bytes` row-major feature
//     row per token, whose linear index h*V + c IS the flat feature index;
//   * this core OWNS flat output columns [slice*cols_per_core, +cols_per_core),
//     i.e. flat feature range [slice*cols_per_core*TILE_W, +cols_per_core*TILE_W)
//     — which, because a token's untilized row is contiguous in flat-feature
//     order, is exactly ONE contiguous `chunk_bytes` slice of every token's row;
//   * so the exchange is: for each of my token rows, send its `d`-th chunk to
//     member `d`'s cb_rm_flat_rows at row offset `t * chunk_bytes`.
//
// Each core therefore ends up holding a [TOKENS_PER_BLOCK, cols_per_core*TILE_W]
// row-major stripe of row stride `chunk_bytes` — precisely the contract the local
// untilize honoured before, which is what op_design.md §1.5's lamp #1 promised.
// Nothing is re-read and nothing is re-written: `o` is split by token, `gate` and
// the output by column, and only the row-major intermediate crosses the NoC.
//
// HELPER SUBSTITUTION, declared up front: mcast_pipe.hpp's SenderPipe /
// ReceiverPipe are NOT usable for this exchange, for three independent reasons
// that are all in the helper's own stated contract:
//   (a) it is a MULTICAST of ONE block to a rectangle at ONE `dst_l1`
//       (`send(src_l1, dst_l1, size)`, "the landing address dst_l1 is identical
//       across all receivers").  Here every destination gets DIFFERENT bytes (its
//       own column chunk) at a DIFFERENT source offset — a scatter, not a
//       broadcast.  There is no block to multicast.
//   (b) its precondition is "single sender per receiver".  This is an ALL-TO-ALL:
//       every member of the group is simultaneously sender to, and receiver from,
//       all `group_cores` members, so the data-ready cell has `group_cores`
//       writers.  Both the Flag and the Counter DataReadySignal are defined for
//       one sender per cell; neither expresses "wait for group_cores distinct
//       contributors".
//   (c) the payload rows are strided on BOTH sides (source stride
//       `local_row_bytes`, destination stride `chunk_bytes`), so even a degenerate
//       1x1 rect could not carry a block's worth in one `send()`.
// What IS used is the layer mcast_pipe itself is built on — the `Noc` and
// `Semaphore<>` object APIs from api/dataflow/noc.h + noc_semaphore.h — rather
// than raw `noc_semaphore_set/wait/inc`.  The two counters are MONOTONE
// (`wait_min` against `(blk+1)*group_cores`, host-initialised to 0, never reset
// by a kernel), which is what makes a member that races ahead into the next block
// harmless: with a set-to-0 reset it could clobber an increment another member had
// not yet observed.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    // CB slot map — injected as preprocessor defines from the ONE host-side
    // source of truth (`_CB_SLOTS` / `_kernel_defines` in
    // onorm_program_descriptor.py).  ONORM_CB_RM_LOCAL *aliases*
    // ONORM_CB_RM_FLAT_ROWS when the block is not split, and every use of both is
    // inside `if constexpr (group_cores > 1)`, so the alias is inert there.
    constexpr uint32_t cb_out_tiles = ONORM_CB_OUT_TILES;
    constexpr uint32_t cb_rm_local = ONORM_CB_RM_LOCAL;
    constexpr uint32_t cb_rm_flat_rows = ONORM_CB_RM_FLAT_ROWS;

    // --- Blocking Model parameters (compile-time; one source of truth on host) ---
    constexpr uint32_t flat_tiles = get_compile_time_arg_val(0);           // FLAT / TILE_W
    constexpr uint32_t cols_per_core = get_compile_time_arg_val(1);        // flat_tiles / group_cores
    constexpr uint32_t tile_rows_per_block = get_compile_time_arg_val(2);  // TOKENS_PER_BLOCK / TILE_H
    constexpr uint32_t blocks_per_batch = get_compile_time_arg_val(3);     // ceil(T / TOKENS_PER_BLOCK)
    constexpr uint32_t token_tile_rows = get_compile_time_arg_val(4);      // Tt = ceil(T / TILE_H)
    constexpr uint32_t dm_block_tiles = get_compile_time_arg_val(5);       // DM_BLOCK_TILES
    constexpr uint32_t page_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t group_cores = get_compile_time_arg_val(7);        // RETILE_GROUP_CORES
    constexpr uint32_t tokens_per_core = get_compile_time_arg_val(8);    // TOKENS_PER_BLOCK / group_cores
    constexpr uint32_t norm_chunk_tokens = get_compile_time_arg_val(9);  // NORM_CHUNK_TOKENS (clamped)
    constexpr uint32_t norm_chunks = get_compile_time_arg_val(10);       // tokens_per_core / norm_chunk_tokens
    constexpr uint32_t v_tiles = get_compile_time_arg_val(11);           // V / TILE_W
    constexpr uint32_t sem_rm_free_id = get_compile_time_arg_val(12);
    constexpr uint32_t sem_rm_data_id = get_compile_time_arg_val(13);

    constexpr auto out_args = TensorAccessorArgs<14>();

    // Derived — never restated literals.
    // Output tiles this core writes per token-block (its column slice of every
    // tile-row of the block); == flat_tiles_per_block at group_cores == 1.
    constexpr uint32_t out_tiles_per_core = tile_rows_per_block * cols_per_core;
    // One token's untilized row-major feature row: v_tiles tiles' worth of
    // row-major bytes = FLAT * sizeof(elem).
    constexpr uint32_t local_row_bytes = v_tiles * page_bytes;
    // The slice of that row one group member owns = cols_per_core * TILE_W elems.
    constexpr uint32_t chunk_bytes = local_row_bytes / group_cores;
    // Pages compute pushes to cb_rm_local per normalize chunk / the writer pops.
    constexpr uint32_t rm_local_chunk_pages = norm_chunk_tokens * v_tiles;
    // This core's whole re-tile stripe, in pages. Exactly cb_rm_flat_rows' size.
    constexpr uint32_t rm_stripe_pages = out_tiles_per_core;

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const uint32_t slice = get_arg_val<uint32_t>(3);

    const auto out_acc = TensorAccessor(out_args, out_addr, page_bytes);

    // ---- cross-core re-tile group state (compiled out entirely at group_cores == 1) ----
    // The group's virtual NoC coords in slice order; every member carries the same
    // list and indexes it by destination slice.
    uint32_t group_x[group_cores];
    uint32_t group_y[group_cores];
    uint32_t rm_base = 0;
    if constexpr (group_cores > 1) {
        for (uint32_t d = 0; d < group_cores; ++d) {
            group_x[d] = get_arg_val<uint32_t>(4 + 2 * d);
            group_y[d] = get_arg_val<uint32_t>(5 + 2 * d);
        }
        // The landing address of every member's stripe.  Safe to capture ONCE and
        // reuse for remote cores because (a) cb_rm_flat_rows is created with the
        // same descriptor on every core of `all_cores`, so it lives at the same L1
        // offset everywhere, and (b) it is EXACTLY one stripe and is fully drained
        // back to its base every block (op_design.md §6.1), so its write pointer
        // is the buffer base at every block boundary.
        rm_base = get_write_ptr(cb_rm_flat_rows);
    }
    [[maybe_unused]] Noc noc;
    [[maybe_unused]] Semaphore<> sem_rm_free(sem_rm_free_id);
    [[maybe_unused]] Semaphore<> sem_rm_data(sem_rm_data_id);

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t bi = start_block + blk;
        const uint32_t b = bi / blocks_per_batch;
        const uint32_t r = bi % blocks_per_batch;

        if constexpr (group_cores > 1) {
            // ===== the cross-core re-tile exchange, one round per token-block =====
            // Monotone target: every member increments each cell exactly once per
            // block, so after block `blk` both cells read (blk+1)*group_cores.
            const uint32_t round = (blk + 1) * group_cores;

            // (a) RECEIVER role, flow control first: claim my stripe (this blocks
            //     until compute has drained the PREVIOUS block's stripe in P7a),
            //     then tell every member their chunks may land.
            {
                MaybeDeviceZoneScope("onorm_rm_open");
                cb_reserve_back(cb_rm_flat_rows, rm_stripe_pages);
                for (uint32_t d = 0; d < group_cores; ++d) {
                    sem_rm_free.up(noc, group_x[d], group_y[d], 1);
                }
                sem_rm_free.wait_min(round);
            }

            // (b) SENDER role: scatter my token rows, one contiguous chunk per
            //     destination per row.  `norm_chunks` groups of
            //     `norm_chunk_tokens * group_cores` writes, ONE barrier per group —
            //     the same "many transfers in flight, one barrier" shape as the
            //     DRAM streams below.
            {
                MaybeDeviceZoneScope("onorm_rm_scatter");
                for (uint32_t chunk = 0; chunk < norm_chunks; ++chunk) {
                    cb_wait_front(cb_rm_local, rm_local_chunk_pages);
                    const uint32_t src_base = get_read_ptr(cb_rm_local);
                    for (uint32_t i = 0; i < norm_chunk_tokens; ++i) {
                        // Row index of this token WITHIN the block's stripe.
                        const uint32_t t = slice * tokens_per_core + chunk * norm_chunk_tokens + i;
                        const uint32_t src_row = src_base + i * local_row_bytes;
                        for (uint32_t d = 0; d < group_cores; ++d) {
                            noc_async_write(
                                src_row + d * chunk_bytes,
                                get_noc_addr(group_x[d], group_y[d], rm_base + t * chunk_bytes),
                                chunk_bytes);
                        }
                    }
                    noc_async_write_barrier();  // ONE barrier for the whole chunk
                    cb_pop_front(cb_rm_local, rm_local_chunk_pages);
                }
            }

            // (c) both roles: my chunks are flushed, so announce them; then wait
            //     for every member's chunks and publish my completed stripe to
            //     compute's tilize (P7a).
            {
                MaybeDeviceZoneScope("onorm_rm_close");
                for (uint32_t d = 0; d < group_cores; ++d) {
                    sem_rm_data.up(noc, group_x[d], group_y[d], 1);
                }
                sem_rm_data.wait_min(round);
                cb_push_back(cb_rm_flat_rows, rm_stripe_pages);
            }
        }

        // ===== drain this core's output column slice to DRAM =====
        // Output shares `gate`'s (T, FLAT) tiling, so the token axis is tile-padded.
        // One consecutive run per tile-row — the same order the reader took `gate`
        // in and P7a's tilize emitted.
        const uint32_t out_row0_tile = (b * token_tile_rows + r * tile_rows_per_block) * flat_tiles;

        MaybeDeviceZoneScope("onorm_write_out");
        for (uint32_t tr = 0; tr < tile_rows_per_block; ++tr) {
            const uint32_t first_tile = out_row0_tile + tr * flat_tiles + slice * cols_per_core;
            uint32_t done = 0;
            while (done < cols_per_core) {
                const uint32_t remaining = cols_per_core - done;
                const uint32_t n = remaining < dm_block_tiles ? remaining : dm_block_tiles;
                cb_wait_front(cb_out_tiles, n);
                const uint32_t l1_read_addr = get_read_ptr(cb_out_tiles);
                for (uint32_t i = 0; i < n; ++i) {
                    noc_async_write(
                        l1_read_addr + i * page_bytes, out_acc.get_noc_addr(first_tile + done + i), page_bytes);
                }
                noc_async_write_barrier();  // ONE barrier for `n` writes
                cb_pop_front(cb_out_tiles, n);
                done += n;
            }
        }
    }
}
