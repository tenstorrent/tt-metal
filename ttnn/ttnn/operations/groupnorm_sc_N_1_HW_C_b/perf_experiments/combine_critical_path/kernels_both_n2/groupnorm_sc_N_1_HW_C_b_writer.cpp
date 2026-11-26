// zone-id pad (make_variants.py)
// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// groupnorm_sc_N_1_HW_C_b writer (BRISC, NoC1).
//
// Per block:
//   gather   : row 0 of the 2 local partial tiles (group lanes of S1 / S2) -> row `slot` of the root's
//              cb_gathered_partials tile pair slot/32, then a semaphore increment on the root.
//   root     : wait for cpi arrivals, publish the gathered tiles to compute; multicast [rstd, mean] from
//              cb_stats_out into every unit core's cb_group_stats (McastArgs; cpi == 1 is a local copy).
//   receiver : take the multicast into cb_group_stats.
//   output   : pass-2 chunks (resident tail first, then the re-read head) -> DRAM via TensorAccessor.
//              SHARDED: compute writes the output shard directly (zero-copy CB); the writer only drains the
//              credits (RM: the head-skip credits compute pushes first, then every chunk).
//
// Unused gather rows (slots >= cpi in the last tile pair) are zeroed once with NoC zeros; senders never
// touch them, so they stay zero across blocks.

#include <cstdint>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {
constexpr uint32_t cb_output_rm = 1;
constexpr uint32_t cb_output_tiles = 5;
constexpr uint32_t cb_local_partials = 15;
constexpr uint32_t cb_gathered_partials = 16;
constexpr uint32_t cb_stats_out = 19;
constexpr uint32_t cb_group_stats = 20;

constexpr uint32_t F32_TILE_BYTES = 4096;
constexpr uint32_t F32_FACE_BYTES = 1024;
constexpr uint32_t F32_FACE_ROW_BYTES = 64;  // 16 fp32 lanes
constexpr uint32_t GATHER_SEMAPHORE_ID = 0;

// Byte offset of tile row `row`'s first half (lanes 0..15) in a 32x32 fp32 tile; the second half
// (lanes 16..31) is one face further.
FORCE_INLINE uint32_t face_row_lo(uint32_t row) {
    return row < 16 ? row * F32_FACE_ROW_BYTES : 2 * F32_FACE_BYTES + (row - 16) * F32_FACE_ROW_BYTES;
}
}  // namespace

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr uint32_t CHUNK_HT = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t WT_TOTAL = get_compile_time_arg_val(3);
    constexpr uint32_t HT = get_compile_time_arg_val(4);
    constexpr uint32_t GX = get_compile_time_arg_val(5);
    constexpr uint32_t OUT_PAGE_BYTES = get_compile_time_arg_val(6);
    constexpr bool IS_ROOT = get_compile_time_arg_val(7) != 0;  // one binary per role (unit root / member)
    constexpr bool IS_RM = get_compile_time_arg_val(8) != 0;
    constexpr bool SHARDED = get_compile_time_arg_val(9) != 0;
    // Per-chunk zones only when the host found every chunk's zone executions inside the marker budget.
    constexpr bool CHUNK_ZONES = get_compile_time_arg_val(10) != 0;
    constexpr auto out_args = TensorAccessorArgs<11>();
    constexpr auto mc = dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("mcast_ct_offset"),
        get_named_compile_time_arg_val("mcast_rt_offset")>();
    using McastT = std::remove_cv_t<decltype(mc)>;

    constexpr uint32_t T = CHUNK_HT * W;
    constexpr uint32_t STATS_BYTES = 2 * F32_TILE_BYTES;

    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t unit_start = get_arg_val<uint32_t>(arg++);
    const uint32_t num_blocks = get_arg_val<uint32_t>(arg++);
    const uint32_t row_start = get_arg_val<uint32_t>(arg++);
    const uint32_t rows_core = get_arg_val<uint32_t>(arg++);
    const uint32_t resident_chunks = get_arg_val<uint32_t>(arg++);
    const uint32_t slot = get_arg_val<uint32_t>(arg++);
    const uint32_t root_x = get_arg_val<uint32_t>(arg++);
    const uint32_t root_y = get_arg_val<uint32_t>(arg++);
    const uint32_t cpi = get_arg_val<uint32_t>(arg++);

    const auto out_acc = TensorAccessor(out_args, out_addr, OUT_PAGE_BYTES);
    Noc noc;

    // CBs are allocated identically on every used core, so the local base is the root's base too.
    const uint32_t gathered_base = get_write_ptr(cb_gathered_partials);
    const uint32_t group_stats_base = get_write_ptr(cb_group_stats);
    const uint32_t sem_l1 = get_semaphore(GATHER_SEMAPHORE_ID);
    const uint64_t root_sem_noc = get_noc_addr(root_x, root_y, sem_l1);

    // Destination of this core's partial row inside the root's gathered tile pair slot/32.
    const uint32_t pair_base = gathered_base + 2 * (slot / 32) * F32_TILE_BYTES;
    const uint32_t row = slot % 32;
    const uint32_t dst_row = pair_base + face_row_lo(row);

    if constexpr (IS_ROOT) {
        const uint32_t first_unused = cpi % 32;
        if (first_unused != 0) {
            // Zero rows first_unused..31 of both tiles of the last pair (the sender-sum matmul adds every row).
            // Per face: faces 0/1 hold rows 0..15, faces 2/3 rows 16..31; zero that face's rows >= first_unused.
            DataflowBuffer gathered_dfb(cb_gathered_partials);
            const uint32_t last_pair = 2 * (R - 1) * F32_TILE_BYTES;
            for (uint32_t face = 0; face < 8; ++face) {  // 2 tiles x 4 faces
                const uint32_t f = face & 3;
                const uint32_t first_row = f < 2 ? first_unused : (first_unused > 16 ? first_unused - 16 : 0);
                if (first_row < 16) {
                    const uint32_t lo = first_row * F32_FACE_ROW_BYTES;
                    noc.async_write_zeros(
                        gathered_dfb,
                        F32_FACE_BYTES - lo,
                        {.offset_bytes = last_pair + (face >> 2) * F32_TILE_BYTES + f * F32_FACE_BYTES + lo});
                }
            }
            noc.write_zeros_l1_barrier();
        }
    }

    const uint32_t num_chunks = (rows_core + CHUNK_HT - 1) / CHUNK_HT;
    const uint32_t K = resident_chunks < num_chunks ? resident_chunks : num_chunks;
    const uint32_t first_resident = num_chunks - K;

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t unit = unit_start + blk;
        const uint32_t n = unit / GX;
        const uint32_t wt0 = (unit % GX) * W;

        // ---------------- gather: local partial row -> root ----------------
        if constexpr (IS_ROOT) {
            cb_reserve_back(cb_gathered_partials, 2 * R);
        }
        {
            // Pass 1 + collapse on this core's compute (the writer idles until the local partials exist).
            MaybeDeviceZoneScope("w_wait_partials");
            cb_wait_front(cb_local_partials, 2);
        }
        {
            MaybeDeviceZoneScope("w_gather_send");
            // Row 0 of each partial tile (lanes 0..15 in face 0, 16..31 in face 1) -> row `slot` of the pair.
            const uint32_t src = get_read_ptr(cb_local_partials);
            for (uint32_t k = 0; k < 4; ++k) {
                const uint32_t off = (k >> 1) * F32_TILE_BYTES + (k & 1) * F32_FACE_BYTES;
                noc_async_write(src + off, get_noc_addr(root_x, root_y, dst_row + off), F32_FACE_ROW_BYTES);
            }
            noc_async_write_barrier();  // the row must land before the root sees the increment
            noc_semaphore_inc(root_sem_noc, 1);
        }
        cb_pop_front(cb_local_partials, 2);

        // ---------------- combine / broadcast ----------------
        if constexpr (IS_ROOT) {
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_l1);
            {
                // The slowest unit core's pass 1 + collapse + gather write.
                MaybeDeviceZoneScope("w_wait_gather");
                noc_semaphore_wait_min(sem, cpi);
            }
            noc_semaphore_set(sem, 0);
            cb_push_back(cb_gathered_partials, 2 * R);

            {
                MaybeDeviceZoneScope("w_wait_finalize");
                cb_wait_front(cb_stats_out, 2);
            }
            cb_reserve_back(cb_group_stats, 2);
            if constexpr (McastT::sender_available) {
                MaybeDeviceZoneScope("w_mcast_send");
                mc.sender(noc).send(get_read_ptr(cb_stats_out), group_stats_base, STATS_BYTES);
            }
            cb_push_back(cb_group_stats, 2);
            cb_pop_front(cb_stats_out, 2);
        } else {
            // Member cores exist only when cpi > 1; their binary carries the receiver pipe only.
            cb_reserve_back(cb_group_stats, 2);
            if constexpr (McastT::receiver_available) {
                // The combine round trip seen from a member: gather -> root finalize -> mcast.
                MaybeDeviceZoneScope("w_mcast_recv");
                mc.receiver(noc).receive();
            }
            cb_push_back(cb_group_stats, 2);
        }

        // ---------------- output: pass-2 order (resident tail first, then the re-read head) ----------------
        MaybeDeviceZoneScope("w_output");
        if constexpr (SHARDED) {
            constexpr uint32_t cb_out = IS_RM ? cb_output_rm : cb_output_tiles;
            const uint32_t credits = (IS_RM ? first_resident : 0) + num_chunks;
            for (uint32_t c = 0; c < credits; ++c) {
                cb_wait_front(cb_out, T);
                cb_pop_front(cb_out, T);
            }
            continue;
        }
        for (uint32_t jj = 0; jj < num_chunks; ++jj) {
            const uint32_t j = jj < K ? first_resident + jj : jj - K;
            const uint32_t row0 = row_start + j * CHUNK_HT;
            const uint32_t rows_left = rows_core - j * CHUNK_HT;
            const uint32_t rows_valid = rows_left < CHUNK_HT ? rows_left : CHUNK_HT;
            {
                MaybeDeviceZoneScopeIf(CHUNK_ZONES, "w_wait_out");
                cb_wait_front(cb_output_tiles, T);
            }
            MaybeDeviceZoneScopeIf(CHUNK_ZONES, "w_issue");
            uint32_t src = get_read_ptr(cb_output_tiles);
            for (uint32_t r = 0; r < rows_valid; ++r) {
                const uint32_t tile0 = (n * HT + row0 + r) * WT_TOTAL + wt0;
                for (uint32_t wt = 0; wt < W; ++wt) {
                    noc_async_write_page(tile0 + wt, out_acc, src);
                    src += OUT_PAGE_BYTES;
                }
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_output_tiles, T);
        }
    }
    MaybeDeviceZoneScope("w_final_barrier");
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
