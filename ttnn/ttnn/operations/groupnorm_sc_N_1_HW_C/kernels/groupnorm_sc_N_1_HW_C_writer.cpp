// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// groupnorm_sc_N_1_HW_C — writer (data movement): the cross-core combine + output store.
//
// Combine scheme (CT arg `combine`, host knob COMBINE_SCHEME — Refinement 5):
//   LOCAL      (P_used == 1) nothing to combine: compute finalizes its own cb_partial; the writer only stores.
//   ALL_GATHER every non-idle core unicasts its record row into EVERY rectangle core's gather tiles (its own
//              included) and then +1 into every other core's gather semaphore; each core waits for P_used - 1
//              increments and hands its own complete gather to compute, which reduces it locally. No root, no serial
//              reduce, no totals broadcast. Idle cores do nothing (their L1 receives the records / increments, which
//              nobody reads). Host-gated to small rectangles (ALL_GATHER_MAX_CORES): the fan-out is O(P_used) commands.
//   ROOT       (Phase-0 scheme, kept as the measured alternative) roles inside an image rectangle (RT arg):
//     ROOT   (p == 0)      zero-fills the PAD rows (p >= P_used) of the gather tiles once, writes its own record,
//                          waits for every rectangle core's semaphore increment, hands the gather to compute,
//                          multicasts the reduced totals (no pre-handshake).
//     MEMBER (0 < p < P_used) writes its partial record into row p of the root's gather tiles, increments the
//                          root's gather semaphore, waits for the totals multicast.
//     IDLE   (p >= P_used) increments the gather semaphore (so the root knows its ReceiverPipe exists) and
//                          receives the totals multicast.
// Then every non-idle core streams cb_out tiles of its own block to DRAM. Blocks are ragged (op_design.md ->
// Work Distribution): compute pushes the nominal `chunk` pages per block with the valid_rows x valid_cols output
// tiles dense at the front; the writer drains the nominal count and stores only the valid tiles.
//
// Why no gather-ready signal / pre-handshake: records only ever touch rows p < P_used and the root only
// ever zeroes rows >= P_used, so the two never overlap and need no ordering. The totals flag cannot be
// clobbered by a late ReceiverPipe constructor (which resets the flag) because every rectangle core
// increments the gather semaphore AFTER constructing its receiver, and the root broadcasts only after all
// increments arrived. Saves two multicast round trips per image versus handshake=True.
//
// Raw dataflow (per op_design.md -> helpers considered and rejected): the record gather is P_used concurrent
// unicasts (ROOT) or multicasts (ALL_GATHER) of 2*Kg non-contiguous 64 B face-row chunks landing at per-sender
// offsets — SenderPipe::send multicasts ONE contiguous block from ONE sender per round, so it does not fit either
// shape; the ROOT totals broadcast does use it. The ALL_GATHER record is a unicast fan-out (see
// send_partial_record_all for why not P_used concurrent multicasts).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "groupnorm_sc_N_1_HW_C_ragged.hpp"

using namespace dataflow_kernel_lib;

namespace {
constexpr uint32_t ROLE_IDLE = 0;
constexpr uint32_t ROLE_MEMBER = 1;
constexpr uint32_t ROLE_ROOT = 2;
// Combine schemes (host: COMBINE_LOCAL / COMBINE_ROOT / COMBINE_ALL_GATHER)
constexpr uint32_t COMBINE_LOCAL = 0;
constexpr uint32_t COMBINE_ROOT = 1;
constexpr uint32_t COMBINE_ALL_GATHER = 2;
}  // namespace

void kernel_main() {
    // ---------------- compile-time args ----------------
    constexpr uint32_t cb_partial = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gather = get_compile_time_arg_val(1);
    constexpr uint32_t cb_totals_src = get_compile_time_arg_val(2);
    constexpr uint32_t cb_totals_recv = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t Kg = get_compile_time_arg_val(5);
    constexpr uint32_t cols = get_compile_time_arg_val(6);
    constexpr uint32_t chunk_rows = get_compile_time_arg_val(7);
    constexpr uint32_t Ht = get_compile_time_arg_val(8);
    constexpr uint32_t Ct = get_compile_time_arg_val(9);
    constexpr uint32_t gather_tiles_per_stat = get_compile_time_arg_val(10);
    constexpr uint32_t sem_gather_id = get_compile_time_arg_val(11);
    constexpr uint32_t out_block = get_compile_time_arg_val(12);  // tiles per store barrier (divides chunk)
    // Per-out_block store sync: flush (writes departed the core -> cb_out pages reusable; one full barrier per
    // image) vs a full DRAM-ack barrier per block. Host knob OUT_STORE_FLUSH_PER_BLOCK.
    constexpr bool store_flush_per_block = get_compile_time_arg_val(13) != 0;
    constexpr uint32_t combine = get_compile_time_arg_val(14);  // COMBINE_LOCAL / COMBINE_ROOT / COMBINE_ALL_GATHER
    static_assert(combine <= COMBINE_ALL_GATHER, "unknown combine scheme");
    constexpr uint32_t MC_CT = 15;
    constexpr uint32_t MC_RT = 18;
    constexpr auto mc = McastArgs<MC_CT, MC_RT>();
    constexpr auto out_args = TensorAccessorArgs<mc.next_compile_time_args_offset()>();

    // ---------------- runtime args ----------------
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t image_begin = get_arg_val<uint32_t>(1);
    const uint32_t image_count = get_arg_val<uint32_t>(2);
    const uint32_t image_stride = get_arg_val<uint32_t>(3);
    const uint32_t row_begin = get_arg_val<uint32_t>(4);
    const uint32_t col_begin = get_arg_val<uint32_t>(5);
    const uint32_t role = get_arg_val<uint32_t>(6);
    const uint32_t p = get_arg_val<uint32_t>(7);
    const uint32_t p_used = get_arg_val<uint32_t>(8);
    const uint32_t root_x = get_arg_val<uint32_t>(9);
    const uint32_t root_y = get_arg_val<uint32_t>(10);
    const uint32_t num_participants = get_arg_val<uint32_t>(11);  // every core of the rectangle (incl. idle)
    const uint32_t Ht_core = get_arg_val<uint32_t>(12);
    const uint32_t Ct_core = get_arg_val<uint32_t>(13);
    // image rectangle corners (virtual NoC coords), valid on every core — McastArgs::rect() is sender-only
    const uint32_t rect_x0 = get_arg_val<uint32_t>(14);
    const uint32_t rect_y0 = get_arg_val<uint32_t>(15);
    const uint32_t rect_x1 = get_arg_val<uint32_t>(16);
    const uint32_t rect_y1 = get_arg_val<uint32_t>(17);

    constexpr uint32_t num_stats = 2 * Kg;
    constexpr uint32_t chunk = chunk_rows * cols;
    constexpr uint32_t out_blocks_per_chunk = chunk / out_block;
    static_assert(chunk % out_block == 0, "out_block must divide the chunk (host derives it so)");
    constexpr uint32_t f32_tile_bytes = get_tile_size(cb_partial);
    constexpr uint32_t gather_tiles = num_stats * gather_tiles_per_stat;
    constexpr uint32_t y_tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t face_bytes = 1024;  // 16x16 Float32 face
    constexpr uint32_t face_row_bytes = 64;

    Noc noc;
    CircularBuffer gather(cb_gather);
    Semaphore<> sem_gather(sem_gather_id);
    const auto out_acc = TensorAccessor(out_args, out_addr, y_tile_bytes);

    // Landing addresses: identical allocation on every core of the rectangle; both CBs have capacity equal
    // to their per-image quantum, so their write pointers return to the base after every image.
    const uint32_t gather_base = gather.get_write_ptr();
    const uint32_t totals_recv_base = get_write_ptr(cb_totals_recv);

    // Ragged accounting (idle cores have Ht_core = Ct_core = 0 and never store).
    const auto row_axis = groupnorm_ragged::split(Ht_core, chunk_rows);
    const auto col_axis = groupnorm_ragged::split(Ct_core, cols);
    const uint32_t num_row_chunks = row_axis.count;
    const uint32_t num_col_groups = col_axis.count;

    // Row 0 of each of this core's 2*Kg partial tiles -> row p of gather tile (p / 32) of the same statistic.
    auto send_partial_record = [&]() {
        cb_wait_front(cb_partial, num_stats);
        const uint32_t src = get_read_ptr(cb_partial);
        const uint32_t gt = p / 32;
        const uint32_t r = p % 32;
        const uint32_t row_face = (r >= 16) ? 2u : 0u;
        for (uint32_t s = 0; s < num_stats; ++s) {
            const uint32_t dst_tile = gt * num_stats + s;
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t src_off = s * f32_tile_bytes + half * face_bytes;
                const uint32_t dst_off =
                    dst_tile * f32_tile_bytes + (row_face + half) * face_bytes + (r & 15) * face_row_bytes;
                noc_async_write(src + src_off, get_noc_addr(root_x, root_y, gather_base + dst_off), face_row_bytes);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_partial, num_stats);
        sem_gather.up(noc, root_x, root_y, 1);
    };

    // cb_out -> DRAM tiles of this core's block. Compute packs the chunk row-major (chunk_rows x cols, one push
    // per tile); the writer drains it in `out_block`-tile groups with ONE sync per group, so the number of
    // tiles in flight per sync is a host knob (OUT_BLOCK_TILES_TARGET) and not an accident of `cols`
    // (which is 1 whenever the ct split gives a core a single tile-column).
    // The per-group sync is `noc_async_writes_flushed` when store_flush_per_block: the block's bytes have left
    // L1 (so its cb_out pages may be popped and re-packed) but the DRAM acks are not awaited, so the core keeps
    // issuing while far-route acks are in flight; ONE `noc_async_write_barrier` per image retires them all.
    // Otherwise a full barrier per group (Phase-0 behaviour; measured 3-4 us per round trip at 110 cores).
    // Ragged blocks: the valid_rows x valid_cols output tiles sit dense at the front of the block's nominal
    // `chunk` pages (idx = r * valid_cols + c); pages idx >= valid carry no data and are only drained.
    auto store_image = [&](uint32_t n) {
        for (uint32_t cg = 0; cg < num_col_groups; ++cg) {
            const uint32_t valid_cols = col_axis.valid(cg, cols);
            for (uint32_t rc = 0; rc < num_row_chunks; ++rc) {
                const uint32_t valid_rows = row_axis.valid(rc, chunk_rows);
                const uint32_t valid = valid_rows * valid_cols;
                // (r, c) walk instead of idx / valid_cols: the divisor is a runtime value here and a per-tile
                // software divide on the RISC is measurable on the latency floor.
                uint32_t idx = 0;
                uint32_t r = row_begin + rc * chunk_rows;
                uint32_t c = 0;
                const uint32_t t0 = col_begin + cg * cols;
                for (uint32_t b = 0; b < out_blocks_per_chunk; ++b) {
                    cb_wait_front(cb_out, out_block);
                    const uint32_t l1 = get_read_ptr(cb_out);
                    for (uint32_t k = 0; k < out_block && idx < valid; ++k, ++idx) {
                        const uint32_t page = n * Ht * Ct + r * Ct + t0 + c;
                        noc_async_write(l1 + k * y_tile_bytes, out_acc.get_noc_addr(page), y_tile_bytes);
                        if (++c == valid_cols) {
                            c = 0;
                            ++r;
                        }
                    }
                    if constexpr (store_flush_per_block) {
                        noc_async_writes_flushed();
                    } else {
                        noc_async_write_barrier();
                    }
                    cb_pop_front(cb_out, out_block);
                }
            }
        }
        if constexpr (store_flush_per_block) {
            noc_async_write_barrier();  // retire this image's stores before the next image / kernel exit
        }
    };

    // Zero the pad rows (cores p >= P_used) of every gather tile ONCE, on each core that reduces a gather (the root
    // under ROOT, every non-idle core under ALL_GATHER). Record rows are never zeroed, so a record that lands before
    // this completes is not disturbed; pad rows are never written afterwards.
    auto zero_gather_pad_rows = [&]() {
        uint32_t issued = 0;
        for (uint32_t gt = 0; gt < gather_tiles_per_stat; ++gt) {
            const uint32_t first_pad_row = (p_used > gt * 32) ? (p_used - gt * 32) : 0u;  // within tile
            if (first_pad_row >= 32) {
                continue;
            }
            for (uint32_t s = 0; s < num_stats; ++s) {
                const uint32_t tile_off = (gt * num_stats + s) * f32_tile_bytes;
                for (uint32_t face = 0; face < 4; ++face) {
                    const uint32_t face_row0 = (face >> 1) * 16;
                    const uint32_t r0 = (first_pad_row > face_row0) ? (first_pad_row - face_row0) : 0u;
                    if (r0 >= 16) {
                        continue;
                    }
                    noc.async_write_zeros(
                        gather,
                        (16 - r0) * face_row_bytes,
                        {.offset_bytes = tile_off + face * face_bytes + r0 * face_row_bytes});
                    ++issued;
                }
            }
        }
        if (issued > 0) {
            noc.write_zeros_l1_barrier();
        }
    };

    // ALL_GATHER record: the same 2*Kg face-row chunks as send_partial_record, UNICAST into every rectangle core's
    // gather tiles (own copy included), then one unicast +1 on every OTHER core's gather semaphore after a write
    // barrier (records landed everywhere; source L1 reusable). Unicast, not multicast: P_used concurrent multicasts
    // into one rectangle on one NoC each path-reserve the whole row ring in the same routing direction and
    // circular-wait — measured as every writer stuck in the write barrier (8x1 and 2x10 rectangles); tt-metal runs
    // one multicast source per rectangle per NoC and mcast_pipe states it as a precondition. The unicast fan-out
    // costs (area - 1) * (2*Kg*2 + 1) NoC commands per core, so the host gates this scheme by rectangle size
    // (ALL_GATHER_MAX_CORES) and larger rectangles keep ROOT.
    // The rectangle is walked in virtual NoC coordinates; Blackhole's virtual worker columns skip 8 and 9 (the same
    // rule McastRect::area() applies), so those two columns are never addressed.
    auto send_partial_record_all = [&](const auto& rect) {
        cb_wait_front(cb_partial, num_stats);
        const uint32_t src = get_read_ptr(cb_partial);
        const uint32_t gt = p / 32;
        const uint32_t r = p % 32;
        const uint32_t row_face = (r >= 16) ? 2u : 0u;
        auto is_worker_col = [](uint32_t x) {
#if defined(ARCH_BLACKHOLE)
            return x != 8u && x != 9u;
#else
            (void)x;
            return true;
#endif
        };
        for (uint32_t y = rect.ylo(); y <= rect.yhi(); ++y) {
            for (uint32_t x = rect.xlo(); x <= rect.xhi(); ++x) {
                if (!is_worker_col(x)) {
                    continue;
                }
                for (uint32_t s = 0; s < num_stats; ++s) {
                    const uint32_t dst_tile = gt * num_stats + s;
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint32_t src_off = s * f32_tile_bytes + half * face_bytes;
                        const uint32_t dst_off =
                            dst_tile * f32_tile_bytes + (row_face + half) * face_bytes + (r & 15) * face_row_bytes;
                        noc_async_write(src + src_off, get_noc_addr(x, y, gather_base + dst_off), face_row_bytes);
                    }
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_partial, num_stats);
        const uint32_t mx = my_x[noc_index];
        const uint32_t my = my_y[noc_index];
        for (uint32_t y = rect.ylo(); y <= rect.yhi(); ++y) {
            for (uint32_t x = rect.xlo(); x <= rect.xhi(); ++x) {
                if (!is_worker_col(x) || (x == mx && y == my)) {
                    continue;
                }
                sem_gather.up(noc, x, y, 1);
            }
        }
    };

    if constexpr (combine == COMBINE_LOCAL) {
        // P_used == 1: compute finalizes its own cb_partial (no gather, no totals); the writer only stores.
        for (uint32_t img = 0; img < image_count; ++img) {
            store_image(image_begin + img * image_stride);
        }
    } else if constexpr (combine == COMBINE_ALL_GATHER) {
        if (role == ROLE_IDLE) {
            return;  // no record, no reduce; the records / increments landing in this core's L1 are never read
        }
        const McastRect<noc_index> rect(rect_x0, rect_y0, rect_x1, rect_y1);
        zero_gather_pad_rows();
        for (uint32_t img = 0; img < image_count; ++img) {
            const uint32_t n = image_begin + img * image_stride;
            // Gather reuse across images is only safe because a multi-image core is alone in its rectangle
            // (host: image_count > 1 => P_used == 1 => LOCAL); a peer's next record could otherwise land here
            // before this core's compute consumed the previous gather.
            gather.reserve_back(gather_tiles);
            send_partial_record_all(rect);
            sem_gather.wait_min((img + 1) * (p_used - 1));  // every other participant's record has landed
            gather.push_back(gather_tiles);
            store_image(n);
        }
    } else if (role == ROLE_ROOT) {
        auto sender = mc.sender(noc);
        zero_gather_pad_rows();
        for (uint32_t img = 0; img < image_count; ++img) {
            const uint32_t n = image_begin + img * image_stride;
            gather.reserve_back(gather_tiles);  // blocks until root compute consumed the previous image
            send_partial_record();
            sem_gather.wait_min((img + 1) * num_participants);
            gather.push_back(gather_tiles);
            // totals: root compute reduced the gather; multicast to the rectangle (self copy when P_n == 1)
            cb_wait_front(cb_totals_src, num_stats);
            cb_reserve_back(cb_totals_recv, num_stats);
            sender.send(get_read_ptr(cb_totals_src), totals_recv_base, num_stats * f32_tile_bytes);
            cb_pop_front(cb_totals_src, num_stats);
            cb_push_back(cb_totals_recv, num_stats);
            store_image(n);
        }
    } else if (role == ROLE_MEMBER) {
        auto receiver = mc.receiver(noc);
        for (uint32_t img = 0; img < image_count; ++img) {
            const uint32_t n = image_begin + img * image_stride;
            send_partial_record();  // ends with the gather semaphore increment (after the receiver exists)
            cb_reserve_back(cb_totals_recv, num_stats);
            receiver.receive();
            cb_push_back(cb_totals_recv, num_stats);
            store_image(n);
        }
    } else {
        auto receiver = mc.receiver(noc);
        for (uint32_t img = 0; img < image_count; ++img) {
            sem_gather.up(noc, root_x, root_y, 1);  // "my receiver exists"; no record
            receiver.receive();
        }
    }
}
