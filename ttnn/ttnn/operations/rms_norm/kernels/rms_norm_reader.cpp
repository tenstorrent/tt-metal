// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm reader (NoC0).
//
//  load_x_block  TILE: x tiles of this core's (R-range x W-slice) -> cb_x, read_chunk_tiles per barrier+push,
//                      one block of lookahead (cb_x holds x_block_depth blocks) so the next block streams in
//                      while compute normalizes / waits for the W_SPLIT combine.
//                TILE sharded: cb_x IS the resident input shard (backed zero-copy) -> publish the block, no read.
//                RM:   sticks of this core's W-slice -> cb_x_rm, one tile-row (32 sticks) per push; from DRAM
//                      (interleaved, TensorAccessor) or core-locally L1 -> L1 from the own shard (sharded: re-stride
//                      from the shard's stick pitch to tilize's core_w_tiles*32 pitch — never an accessor read).
//  wmask         sharded TILE + non-aligned W: the owner of the partial-W tile fills the 0/1 lane mask the
//                collapse reduce applies to that tile's squares (the shard's pad lanes are never written).
//  W_SPLIT root: gather rendezvous -> push cb_gathered (partials land by remote writes from every line core).
//  W_SPLIT all:  rinv receive -> push cb_rinv (non-root: Mcast receiver; root: its own writer's loopback send,
//                signalled through the local SEM_RINV_LOCAL counter — the Mcast sender holds no receiver role).
//
// Raw NoC (no kernel_lib dataflow helper reads tile ranges / nominal-stride stick slices; op_design.md
// "Helpers considered and rejected").

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

constexpr uint32_t cb_x_rm = 0;
constexpr uint32_t cb_x = 1;
constexpr uint32_t cb_gathered = 6;
constexpr uint32_t cb_rinv = 9;
constexpr uint32_t cb_wmask = 12;
constexpr uint32_t cb_reduce_scaler = 15;
constexpr uint32_t cb_jsel = 17;

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;

// Zero the W-padding lanes [w_valid, 32) of one tile (all 32 rows, face layout) at L1 address `tile_addr`.
template <uint32_t es>
FORCE_INLINE void zero_w_pad_lanes(uint32_t tile_addr, uint32_t w_valid) {
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * es;
    constexpr uint32_t face_row_bytes = FACE_DIM * es;
    for (uint32_t face = 0; face < 4; ++face) {
        const uint32_t col_lo = (face & 1) ? FACE_DIM : 0;
        if (w_valid >= col_lo + FACE_DIM) {
            continue;  // face fully valid
        }
        const uint32_t start_col = (w_valid > col_lo) ? (w_valid - col_lo) : 0;
        const uint32_t face_addr = tile_addr + face * face_bytes;
        for (uint32_t r = 0; r < FACE_DIM; ++r) {
            fill_l1_range<es>(face_addr + r * face_row_bytes + start_col * es, (FACE_DIM - start_col) * es, 0);
        }
    }
}

void kernel_main() {
    constexpr bool is_rm = get_compile_time_arg_val(0) != 0;
    constexpr bool is_wsplit = get_compile_time_arg_val(1) != 0;
    constexpr uint32_t block_rows = get_compile_time_arg_val(2);
    constexpr uint32_t core_w_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t read_chunk_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t x_block_depth = get_compile_time_arg_val(5);
    constexpr uint32_t x_rm_depth = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t es = get_compile_time_arg_val(8);
    constexpr uint32_t w_valid_last = get_compile_time_arg_val(9);
    constexpr uint32_t x_page_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t Wt = get_compile_time_arg_val(11);
    constexpr uint32_t w_split = get_compile_time_arg_val(12);
    constexpr uint32_t sem_partials_id = get_compile_time_arg_val(13);
    constexpr uint32_t sem_rinv_local_id = get_compile_time_arg_val(14);
    constexpr bool x_sharded = get_compile_time_arg_val(15) != 0;
    constexpr bool mask_partial = get_compile_time_arg_val(16) != 0;
    constexpr bool x_resident = x_sharded && !is_rm;
    constexpr uint32_t w_chunk_tiles = get_compile_time_arg_val(17);  // RM staging chunk (== core_w_tiles: none)
    constexpr uint32_t num_w_chunks = core_w_tiles / w_chunk_tiles;
    static_assert(!is_rm || num_w_chunks * w_chunk_tiles == core_w_tiles, "RM pitch is a chunk multiple");
    constexpr bool collapse_reduce_tile = get_compile_time_arg_val(18) != 0;
    // Dense rinv: Σx² / rinv travel as one dense tile per dense_group tile-rows; this kernel generates the
    // one-hot column selectors (cb_jsel) the densify / expand matmuls use.
    constexpr bool dense = get_compile_time_arg_val(19) != 0;
    constexpr uint32_t jsel_tiles = get_compile_time_arg_val(20);
    constexpr uint32_t int_tile_bytes = get_compile_time_arg_val(21);
    constexpr uint32_t dense_group = TILE_DIM;
    constexpr auto x_args = TensorAccessorArgs<22>();
    // Block-float input (bf8b: tile_bytes != 32*32*es) has no per-element addressing: the RM stick path and the
    // pad-lane zeroing must never be reached for it (host guard in create_program_descriptor).
    constexpr bool is_block_float = tile_bytes != TILE_DIM * TILE_DIM * es;
    static_assert(!is_block_float || (!is_rm && w_valid_last == TILE_DIM), "block-float input: TILE + aligned W only");
    constexpr auto mcast = McastArgs<
        get_named_compile_time_arg_val("rinv_mcast_ct_offset"),
        get_named_compile_time_arg_val("rinv_mcast_rt_offset")>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t line_rows = get_arg_val<uint32_t>(2);
    const uint32_t w_start = get_arg_val<uint32_t>(3);
    const uint32_t w_valid = get_arg_val<uint32_t>(4);
    const uint32_t w_valid_elems = get_arg_val<uint32_t>(5);
    const uint32_t owns_partial_last = get_arg_val<uint32_t>(6);
    const uint32_t is_root = get_arg_val<uint32_t>(7);
    const uint32_t stick_start = get_arg_val<uint32_t>(8);
    const uint32_t stick_end = get_arg_val<uint32_t>(9);
    const uint32_t x_col_offset = get_arg_val<uint32_t>(10);  // interleaved RM: slice byte offset in a stick

    const auto x_acc = TensorAccessor(x_args, x_addr, x_page_bytes);
    Noc noc;
    const uint32_t num_blocks = (line_rows + block_rows - 1) / block_rows;

    // ---- one-time pre-zero of never-written L1 (padding stays 0 forever; pushes are nominal) ----
    if constexpr (is_rm) {
        // Stick reads cover only [0, valid_bytes) of each stick slot: W padding and tail tiles stay zero.
        CircularBuffer x_rm_buf(cb_x_rm);
        noc.async_write_zeros(x_rm_buf, x_rm_depth * w_chunk_tiles * tile_bytes);
        noc.write_zeros_l1_barrier();
    } else if constexpr (!x_resident) {
        if (w_valid < core_w_tiles) {
            // Nominal tail tiles (never read) of every block slot of cb_x.
            CircularBuffer x_buf(cb_x);
            for (uint32_t slot = 0; slot < x_block_depth * block_rows; ++slot) {
                noc.async_write_zeros(
                    x_buf,
                    (core_w_tiles - w_valid) * tile_bytes,
                    {.offset_bytes = (slot * core_w_tiles + w_valid) * tile_bytes});
            }
            noc.write_zeros_l1_barrier();
        }
    }

    // Sharded TILE + non-aligned W: the 0/1 mask for the partial-W tile (once; compute never pops it until the end).
    if constexpr (mask_partial) {
        if (owns_partial_last) {
            prepare_reduce_mask<cb_wmask, ReduceDim::REDUCE_ROW>(w_valid_last);
        }
    }

    // Dense rinv selectors J_r (r < jsel_tiles): 1.0 in column r, 0 elsewhere. Column r lies in faces 0 and 2
    // (r < 16) or 1 and 3 (r >= 16), at face column r % 16, so a selector is two copies of the one-hot face F_c
    // (16 rows, each with a 1.0 at column c). Zero everything over the NoC, write F_c into face 0 of J_c (c < 16:
    // 16 stores each), then assemble every other face with local 512 B NoC copies of those faces. Issued after the
    // first x block is published so it overlaps compute's sumsq (it is first read by the densify).
    auto gen_jsel = [&]() {
        constexpr uint32_t es_int = int_tile_bytes / (TILE_DIM * TILE_DIM);
        static_assert(!dense || es_int == 2 || es_int == 4, "selector tiles are Float16_b or Float32");
        constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * es_int;
        CircularBuffer jsel_buf(cb_jsel);
        cb_reserve_back(cb_jsel, jsel_tiles);
        noc.async_write_zeros(jsel_buf, jsel_tiles * int_tile_bytes);
        noc.write_zeros_l1_barrier();
        const uint32_t base = get_write_ptr(cb_jsel);
        const uint32_t n_lo = jsel_tiles < FACE_DIM ? jsel_tiles : FACE_DIM;
        for (uint32_t c = 0; c < n_lo; ++c) {
            const uint32_t face0 = base + c * int_tile_bytes;
            for (uint32_t i = 0; i < FACE_DIM; ++i) {
                const uint32_t a = face0 + (i * FACE_DIM + c) * es_int;
                if constexpr (es_int == 2) {
                    *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(a) = 0x3F80;  // bf16 1.0
                } else {
                    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a) = 0x3F800000;  // fp32 1.0
                }
            }
        }
        for (uint32_t r = 0; r < jsel_tiles; ++r) {
            const uint32_t src = base + (r % FACE_DIM) * int_tile_bytes;  // F_{r % 16} (face 0 of J_{r % 16})
            const uint32_t dst = base + r * int_tile_bytes;
            if (r < FACE_DIM) {
                noc_async_read(get_noc_addr(src), dst + 2 * face_bytes, face_bytes);  // face 2
            } else {
                noc_async_read(get_noc_addr(src), dst + face_bytes, face_bytes);  // face 1
                noc_async_read(get_noc_addr(src), dst + 3 * face_bytes, face_bytes);  // face 3
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_jsel, jsel_tiles);
    };

    auto load_x_block = [&](uint32_t block_idx) {
        const uint32_t row0 = block_idx * block_rows;
        const uint32_t rows = (line_rows - row0 < block_rows) ? (line_rows - row0) : block_rows;
        if constexpr (is_rm) {
            // One staging push per (tile-row, W-chunk): 32 sticks x w_chunk_tiles*32 elements, in tile-row-major
            // chunk order (what tilize<w_chunk_tiles>(rows * num_w_chunks) consumes). Unchunked: one push per row.
            constexpr uint32_t chunk_elems = w_chunk_tiles * TILE_DIM;
            constexpr uint32_t stick_stride = chunk_elems * es;
            CircularBuffer x_rm_slot(cb_x_rm);
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t stick0 = stick_start + (row0 + r) * TILE_DIM;
                for (uint32_t j = 0; j < num_w_chunks; ++j) {
                    const uint32_t e0 = j * chunk_elems;
                    const uint32_t valid = (w_valid_elems > e0) ? ((w_valid_elems - e0 < chunk_elems)
                                                                       ? (w_valid_elems - e0)
                                                                       : chunk_elems)
                                                                : 0;
                    cb_reserve_back(cb_x_rm, w_chunk_tiles);
                    const uint32_t l1 = get_write_ptr(cb_x_rm);
                    if constexpr (num_w_chunks > 1) {
                        // Slots rotate across chunks: a short chunk must not inherit a full chunk's stale tail.
                        if (valid < chunk_elems) {
                            noc.async_write_zeros(x_rm_slot, w_chunk_tiles * tile_bytes);
                            noc.write_zeros_l1_barrier();
                        }
                    }
                    if (valid > 0) {
                        for (uint32_t s = 0; s < TILE_DIM && stick0 + s < stick_end; ++s) {
                            uint64_t src;
                            if constexpr (x_sharded) {
                                // own shard, core-local: stick pitch = shard page (x_page_bytes)
                                src = get_noc_addr(x_addr + (stick0 + s - stick_start) * x_page_bytes + e0 * es);
                            } else {
                                src = x_acc.get_noc_addr(stick0 + s, x_col_offset + e0 * es);
                            }
                            noc_async_read(src, l1 + s * stick_stride, valid * es);
                        }
                        noc_async_read_barrier();
                    }
                    cb_push_back(cb_x_rm, w_chunk_tiles);
                }
            }
        } else if constexpr (x_resident) {
            // The block is already resident: cb_x is backed on the input shard. Publish it (no NoC traffic).
            cb_reserve_back(cb_x, rows * core_w_tiles);
            cb_push_back(cb_x, rows * core_w_tiles);
        } else {
            const uint32_t block_tiles = rows * core_w_tiles;
            for (uint32_t i0 = 0; i0 < block_tiles; i0 += read_chunk_tiles) {
                const uint32_t n = (block_tiles - i0 < read_chunk_tiles) ? (block_tiles - i0) : read_chunk_tiles;
                cb_reserve_back(cb_x, n);
                const uint32_t l1 = get_write_ptr(cb_x);
                for (uint32_t j = 0; j < n; ++j) {
                    const uint32_t i = i0 + j;
                    const uint32_t r = i / core_w_tiles;
                    const uint32_t c = i - r * core_w_tiles;
                    if (c < w_valid) {
                        const uint32_t page = (row_start + row0 + r) * Wt + w_start + c;
                        noc_async_read(x_acc.get_noc_addr(page), l1 + j * tile_bytes, tile_bytes);
                    }
                }
                noc_async_read_barrier();
                if constexpr (w_valid_last < TILE_DIM) {
                    if (owns_partial_last) {
                        for (uint32_t j = 0; j < n; ++j) {
                            const uint32_t i = i0 + j;
                            const uint32_t c = i % core_w_tiles;
                            if (c == w_valid - 1) {
                                zero_w_pad_lanes<es>(l1 + j * tile_bytes, w_valid_last);
                            }
                        }
                    }
                }
                cb_push_back(cb_x, n);
            }
        }
    };

    // TILE: one block of lookahead (cb_x depth); RM: none (cb_x_rm holds only tile-rows of the current block and
    // compute cannot tilize block b+1 before untilizing block b, so a lookahead read would starve the rinv receive).
    constexpr uint32_t lookahead = is_rm ? 0 : 1;

    static_assert(is_rm || x_block_depth >= 2, "TILE lookahead needs cb_x to hold two blocks");

    Semaphore<> sem_partials(sem_partials_id);
    Semaphore<> sem_rinv_local(sem_rinv_local_id);

    // Constant tiles compute first reads after block 0's sumsq: built once block 0's x is on its way (they would
    // otherwise sit in front of it): the ReduceTile collapse's ones scaler, the dense selectors.
    auto gen_constants = [&]() {
        if constexpr (collapse_reduce_tile) {
            calculate_and_prepare_reduce_scaler<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>();
        }
        if constexpr (dense) {
            gen_jsel();
        }
    };

    for (uint32_t b = 0; b < lookahead && b < num_blocks; ++b) {
        load_x_block(b);
    }
    if constexpr (lookahead > 0) {
        if (num_blocks > 0) {
            gen_constants();  // block 0 already published
        }
    }
    for (uint32_t b = 0; b < num_blocks; ++b) {
        if (b + lookahead < num_blocks) {
            load_x_block(b + lookahead);
        }
        if constexpr (lookahead == 0) {
            if (b == 0) {
                gen_constants();  // right after block 0's x
            }
        }
        if constexpr (is_wsplit) {
            const uint32_t row0 = b * block_rows;
            const uint32_t rows = (line_rows - row0 < block_rows) ? (line_rows - row0) : block_rows;
            // partial / rinv tiles of this block: one per tile-row (sparse) or per dense group (dense)
            const uint32_t units = dense ? (rows + dense_group - 1) / dense_group : rows;
            if (is_root) {
                // gather rendezvous: every line core (root included) wrote its partial(b) into cb_gathered.
                cb_reserve_back(cb_gathered, units * w_split);
                sem_partials.wait_min((b + 1) * w_split);
                cb_push_back(cb_gathered, units * w_split);
                // rinv(b): delivered into this core's cb_rinv by its own writer's loopback multicast.
                cb_reserve_back(cb_rinv, units);
                sem_rinv_local.wait_min(b + 1);
                cb_push_back(cb_rinv, units);
            } else {
                cb_reserve_back(cb_rinv, units);
                auto receiver = mcast.optional_receiver(noc);
                ASSERT(receiver.has_value());
                if (receiver) {
                    receiver->receive(b);
                }
                cb_push_back(cb_rinv, units);
            }
        }
    }
}
