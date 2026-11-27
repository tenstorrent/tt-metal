// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// groupnorm_sc_N_1_HW_C_b writer (BRISC; NoC1, NoC0 on the DRAM odd-column cores: program descriptor "NoC pairing").
//
// Per block:
//   gather   : row 0 of the 2 local partial tiles (group lanes of S1 / S2) -> row `slot` of the root's
//              cb_gathered_partials tile pair slot/32, then a semaphore increment on the root. SHARDED: one pair
//              per image of the unit (R pairs each); a core sends each of its image segments' partials to that
//              image's pairs and zero rows (from MEM_ZEROS) to the unit images it holds no rows of.
//   root     : wait for cpi arrivals, publish the gathered tiles to compute; multicast [rstd, mean] per image from
//              cb_stats_out into every unit core's cb_group_stats (McastArgs; cpi == 1 is a local copy).
//   receiver : take the multicast into cb_group_stats.
//   output   : this RISC's share of the pass-2 chunks -> DRAM via TensorAccessor. Compute packs each chunk for
//              whichever DM RISC has a free output slot (this one's cb_output_tiles preferred, else the reader's
//              cb_output_b) and names its pass-2 index q (resident tail first, then the re-read head) on
//              cb_out_seq_a here; END closes the block. The reader writes the other share on its NoC.
//              SHARDED: compute writes the output shard directly (zero-copy CB); the writer only drains the
//              credits (RM: the head-skip credits compute pushes first, then every chunk), and (RM ragged) copies
//              each staged tile row's valid sticks from cb_rm_stage_out back into the output shard.
//
// Unused gather rows (slots >= cpi in the last tile pair) are zeroed once with NoC zeros; senders never
// touch them, so they stay zero across blocks.

#include <cstdint>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "vendor/gn_perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/operations/groupnorm_sc_N_1_HW_C_b/device/kernels/groupnorm_sc_N_1_HW_C_b_args.hpp"

namespace {
namespace gcb = gn_sc_args::cb;
constexpr uint32_t cb_output_rm = gcb::OUTPUT_RM;
constexpr uint32_t cb_output_tiles = gcb::OUTPUT_TILES;
constexpr uint32_t cb_local_partials = gcb::LOCAL_PARTIALS;
constexpr uint32_t cb_gathered_partials = gcb::GATHERED_PARTIALS;
constexpr uint32_t cb_stats_out = gcb::STATS_OUT;
constexpr uint32_t cb_group_stats = gcb::GROUP_STATS;
constexpr uint32_t cb_rm_stage_out = gcb::RM_STAGE_OUT;  // sharded RM: normalized staged tile rows (untilized)
constexpr uint32_t cb_out_seq_a = gcb::OUT_SEQ_A;  // DRAM interleaved: pass-2 index of each chunk in cb_output_tiles
constexpr uint32_t OUT_SEQ_END = gn_sc_args::OUT_SEQ_END;

constexpr uint32_t F32_TILE_BYTES = 4096;
constexpr uint32_t F32_FACE_BYTES = 1024;
constexpr uint32_t F32_FACE_ROW_BYTES = 64;  // 16 fp32 lanes
constexpr uint32_t GATHER_SEMAPHORE_ID = gn_sc_args::GATHER_SEMAPHORE_ID;

// Byte offset of tile row `row`'s first half (lanes 0..15) in a 32x32 fp32 tile; the second half
// (lanes 16..31) is one face further.
FORCE_INLINE uint32_t face_row_lo(uint32_t row) {
    return row < 16 ? row * F32_FACE_ROW_BYTES : 2 * F32_FACE_BYTES + (row - 16) * F32_FACE_ROW_BYTES;
}
}  // namespace

void kernel_main() {
    namespace wct = gn_sc_args::writer_ct;
    constexpr uint32_t W = get_compile_time_arg_val(wct::W);
    constexpr uint32_t CHUNK_HT = get_compile_time_arg_val(wct::CHUNK_HT);
    constexpr uint32_t R = get_compile_time_arg_val(wct::R);
    constexpr uint32_t WT_TOTAL = get_compile_time_arg_val(wct::WT_TOTAL);
    constexpr uint32_t HT = get_compile_time_arg_val(wct::HT);
    constexpr uint32_t GX = get_compile_time_arg_val(wct::GX);
    constexpr uint32_t OUT_PAGE_BYTES = get_compile_time_arg_val(wct::OUT_PAGE_BYTES);
    constexpr bool IS_ROOT = get_compile_time_arg_val(wct::IS_ROOT) != 0;  // one binary per role (root / member)
    constexpr bool IS_RM = get_compile_time_arg_val(wct::IS_RM) != 0;
    constexpr bool SHARDED = get_compile_time_arg_val(wct::SHARDED) != 0;
    // Per-chunk zones only when the host found every chunk's zone executions inside the marker budget.
    constexpr bool CHUNK_ZONES = get_compile_time_arg_val(wct::CHUNK_ZONES) != 0;
    constexpr uint32_t NU = get_compile_time_arg_val(wct::NU);  // images per statistic unit (2 stats pages each)
    constexpr bool HAS_STAGE = get_compile_time_arg_val(wct::HAS_STAGE) != 0;
    constexpr bool OPS = get_compile_time_arg_val(wct::OPS) != 0;      // sharded, fully resident pieces
    constexpr uint32_t P = get_compile_time_arg_val(wct::MAP_PERIOD);  // sharded RM: the stick's channels
    // Refinement 6 (interleaved, C % 32 != 0): the last column slice's valid channels -> its W_LAST output tiles.
    constexpr uint32_t VC_LAST = get_compile_time_arg_val(wct::VC_LAST);
    constexpr uint32_t W_LAST = (VC_LAST + 31) / 32;
    constexpr auto out_args = TensorAccessorArgs<wct::COUNT>();
    constexpr auto mc = dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("mcast_ct_offset"),
        get_named_compile_time_arg_val("mcast_rt_offset")>();
    using McastT = std::remove_cv_t<decltype(mc)>;

    constexpr uint32_t T = CHUNK_HT * W;
    constexpr uint32_t PAIR_BYTES = 2 * F32_TILE_BYTES;
    // cb_group_stats pages: [rstd, mean*rstd] per image, then e; only the image pairs are multicast.
    constexpr uint32_t N_STATS = 2 * NU + 1;

    namespace wrt = gn_sc_args::writer_rt;
    const uint32_t out_addr = get_arg_val<uint32_t>(wrt::OUT_ADDR);
    const uint32_t unit_start = get_arg_val<uint32_t>(wrt::UNIT_START);
    const uint32_t num_blocks = get_arg_val<uint32_t>(wrt::NUM_BLOCKS);
    const uint32_t row_start = get_arg_val<uint32_t>(wrt::ROW_START);
    const uint32_t rows_core = get_arg_val<uint32_t>(wrt::ROWS_CORE);
    const uint32_t resident_chunks = get_arg_val<uint32_t>(wrt::RESIDENT_CHUNKS);
    const uint32_t slot = get_arg_val<uint32_t>(wrt::SLOT);
    const uint32_t root_x = get_arg_val<uint32_t>(wrt::ROOT_X);
    const uint32_t root_y = get_arg_val<uint32_t>(wrt::ROOT_Y);
    const uint32_t cpi = get_arg_val<uint32_t>(wrt::CPI);
    // NU == 1 (every unit holds one image: all even shards, DRAM) folds the per-image loops to their constant trip.
    const uint32_t n_img_rt = get_arg_val<uint32_t>(wrt::N_IMG);        // images of this core's unit (<= NU)
    const uint32_t n_seg_rt = get_arg_val<uint32_t>(wrt::N_SEG);        // this core's image segments
    const uint32_t seg_img0_rt = get_arg_val<uint32_t>(wrt::SEG_IMG0);  // unit image of its first segment
    const uint32_t n_img = NU == 1 ? 1 : n_img_rt;
    const uint32_t n_seg = NU == 1 ? 1 : n_seg_rt;
    const uint32_t seg_img0 = NU == 1 ? 0 : seg_img0_rt;
    const uint32_t rm_credit_pages = get_arg_val<uint32_t>(wrt::RM_CREDIT_PAGES);
    const uint32_t n_staged = get_arg_val<uint32_t>(wrt::N_STAGED);  // RM staged rows: (shard stick, position, sticks)
    const uint32_t staged_arg = wrt::STAGED_TAIL;

    const auto out_acc = TensorAccessor(out_args, out_addr, OUT_PAGE_BYTES);
    Noc noc;

    // CBs are allocated identically on every used core, so the local base is the root's base too.
    const uint32_t gathered_base = get_write_ptr(cb_gathered_partials);
    const uint32_t group_stats_base = get_write_ptr(cb_group_stats);
    const uint32_t sem_l1 = get_semaphore(GATHER_SEMAPHORE_ID);
    const uint64_t root_sem_noc = get_noc_addr(root_x, root_y, sem_l1);

    // Destination of this core's partial row inside the root's gathered tile pair slot/32 (of unit image 0; image i
    // is R pairs further).
    const uint32_t dst_row = gathered_base + (slot / 32) * PAIR_BYTES + face_row_lo(slot % 32);

    {
        // e = 1.0 (page 2*NU of cb_group_stats; only column 0 counts, the rest must be finite). Written once: the CB
        // moves N_STATS pages per block, so that page is always these bytes.
        volatile tt_l1_ptr uint32_t* e =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(group_stats_base + NU * PAIR_BYTES);
        for (uint32_t i = 0; i < F32_TILE_BYTES / 4; ++i) {
            e[i] = 0x3F800000u;
        }
    }
    if constexpr (IS_ROOT) {
        const uint32_t first_unused = cpi % 32;
        if (first_unused != 0) {
            // Zero rows first_unused..31 of both tiles of the last pair (the sender-sum matmul adds every row).
            // Per face: faces 0/1 hold rows 0..15, faces 2/3 rows 16..31; zero that face's rows >= first_unused.
            DataflowBuffer gathered_dfb(cb_gathered_partials);
            for (uint32_t i = 0; i < n_img; ++i) {
                const uint32_t last_pair = (i * R + R - 1) * PAIR_BYTES;
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
        const uint32_t w_blk = unit % GX == GX - 1 ? W_LAST : W;  // the slice's tiles in the output row

        // ---------------- gather: local partial row(s) -> root ----------------
        if constexpr (IS_ROOT) {
            cb_reserve_back(cb_gathered_partials, 2 * R * NU);
        }
        {
            // Pass 1 + collapse on this core's compute (the writer idles until the local partials exist).
            MaybeDeviceZoneScope("w_wait_partials");
            cb_wait_front(cb_local_partials, 2 * n_seg);
        }
        {
            MaybeDeviceZoneScope("w_gather_send");
            // Row 0 of each partial tile (lanes 0..15 in face 0, 16..31 in face 1) -> row `slot` of the image's pair;
            // zeros (MEM_ZEROS) for the unit images this core holds no rows of.
            const uint32_t src0 = get_read_ptr(cb_local_partials);
            for (uint32_t i = 0; i < n_img; ++i) {
                const bool own = i >= seg_img0 && i < seg_img0 + n_seg;
                const uint32_t src = own ? src0 + (i - seg_img0) * PAIR_BYTES : MEM_ZEROS_BASE;
                const uint32_t dst = dst_row + i * R * PAIR_BYTES;
                for (uint32_t k = 0; k < 4; ++k) {
                    const uint32_t off = (k >> 1) * F32_TILE_BYTES + (k & 1) * F32_FACE_BYTES;
                    noc_async_write(own ? src + off : src, get_noc_addr(root_x, root_y, dst + off), F32_FACE_ROW_BYTES);
                }
            }
            noc_async_write_barrier();  // the rows must land before the root sees the increment
            noc_semaphore_inc(root_sem_noc, 1);
        }
        cb_pop_front(cb_local_partials, 2 * n_seg);

        // ---------------- combine / broadcast ----------------
        if constexpr (IS_ROOT) {
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_l1);
            {
                // The slowest unit core's pass 1 + collapse + gather write.
                MaybeDeviceZoneScope("w_wait_gather");
                noc_semaphore_wait_min(sem, cpi);
            }
            noc_semaphore_set(sem, 0);
            cb_push_back(cb_gathered_partials, 2 * R * NU);

            {
                MaybeDeviceZoneScope("w_wait_finalize");
                cb_wait_front(cb_stats_out, 2 * n_img);
            }
            cb_reserve_back(cb_group_stats, N_STATS);
            if constexpr (McastT::sender_available) {
                MaybeDeviceZoneScope("w_mcast_send");
                mc.sender(noc).send(get_read_ptr(cb_stats_out), group_stats_base, n_img * PAIR_BYTES);
            }
            cb_push_back(cb_group_stats, N_STATS);
            cb_pop_front(cb_stats_out, 2 * n_img);
        } else {
            // Member cores exist only when cpi > 1; their binary carries the receiver pipe only.
            cb_reserve_back(cb_group_stats, N_STATS);
            if constexpr (McastT::receiver_available) {
                // The combine round trip seen from a member: gather -> root finalize -> mcast.
                MaybeDeviceZoneScope("w_mcast_recv");
                mc.receiver(noc).receive();
            }
            cb_push_back(cb_group_stats, N_STATS);
        }

        // ---------------- output: pass-2 order (resident tail first, then the re-read head) ----------------
        MaybeDeviceZoneScope("w_output");
        if constexpr (SHARDED) {
            constexpr uint32_t cb_out = IS_RM ? cb_output_rm : cb_output_tiles;
            if constexpr (HAS_STAGE) {
                // Staged tile rows: compute untilizes each into cb_rm_stage_out (in op order, between the shard's
                // own tile rows); its valid sticks go back to the output shard by a local NoC copy.
                constexpr uint32_t STICK = P * 2;
                for (uint32_t i = 0; i < n_staged; ++i) {
                    const uint32_t dst = get_arg_val<uint32_t>(staged_arg + 3 * i) * STICK;
                    const uint32_t pos = get_arg_val<uint32_t>(staged_arg + 3 * i + 1) * STICK;
                    const uint32_t len = get_arg_val<uint32_t>(staged_arg + 3 * i + 2) * STICK;
                    cb_wait_front(cb_rm_stage_out, W);
                    noc_async_write(get_read_ptr(cb_rm_stage_out) + pos, get_noc_addr(out_addr + dst), len);
                    noc_async_write_barrier();
                    cb_pop_front(cb_rm_stage_out, W);
                }
            }
            // The shard's own pieces (compute packs them in place): drain the credits.
            if constexpr (IS_RM && !OPS) {
                // Partial residency: the head-skip credits wrap the shard FIFO, so drain chunk by chunk.
                for (uint32_t c = 0; c < first_resident + num_chunks; ++c) {
                    cb_wait_front(cb_out, T);
                    cb_pop_front(cb_out, T);
                }
            } else {
                const uint32_t pages = IS_RM ? rm_credit_pages : num_chunks * T;
                if (pages > 0) {
                    cb_wait_front(cb_out, pages);
                    cb_pop_front(cb_out, pages);
                }
            }
            continue;
        }
        for (;;) {
            cb_wait_front(cb_out_seq_a, 1);
            invalidate_l1_cache();
            const uint32_t jj = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_out_seq_a));
            cb_pop_front(cb_out_seq_a, 1);
            if (jj == OUT_SEQ_END) {
                break;
            }
            const uint32_t j = jj < K ? first_resident + jj : jj - K;
            const uint32_t row0 = row_start + j * CHUNK_HT;
            const uint32_t rows_left = rows_core - j * CHUNK_HT;
            const uint32_t rows_valid = rows_left < CHUNK_HT ? rows_left : CHUNK_HT;
            {
                MaybeDeviceZoneScopeIf(CHUNK_ZONES, "w_wait_out");
                cb_wait_front(cb_output_tiles, T);
            }
            MaybeDeviceZoneScopeIf(CHUNK_ZONES, "w_issue");
            const uint32_t src0 = get_read_ptr(cb_output_tiles);
            for (uint32_t r = 0; r < rows_valid; ++r) {
                const uint32_t tile0 = (n * HT + row0 + r) * WT_TOTAL + wt0;
                uint32_t src = src0 + r * W * OUT_PAGE_BYTES;
                for (uint32_t wt = 0; wt < w_blk; ++wt) {
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
