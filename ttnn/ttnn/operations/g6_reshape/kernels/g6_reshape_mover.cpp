// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Perf 1 (planfree_tile_mover, lean_pre_all). Two realizations, chosen at compile time by the geometry:
//  - kStoreDirect (tile_face_row with H_out == 1: one logical row per output tile, so every output face-row is
//    its own write even through the image): plan-free. Sources are closed form per (segment, half) with
//    add-only steps; read runs coalesce consecutive input face-rows; the face-rows are written straight from
//    staging (no plan, no permute copy, no cache invalidate / fence). The write list is built between the read
//    issue and the read barrier, in the DRAM read-latency shadow, and issued in a tight loop after it.
//  - image path (every other geometry): the move-plan walk below, with the merged-run store list likewise
//    built before the read barrier and issued in a tight loop after the permute.
// q553 (1,1,6,256)->(1,1,1536) bf16: 1735 -> ~1090 ns. The two run lambdas MUST stay always_inline (GCC
// outlined them and that alone cost ~300 ns on q553). No raw LLK: dataflow_api NoC calls + TensorAccessor.
//
// One g6_reshape mover. A mover owns a contiguous range of output atoms (atom_rows output rows of one
// output image, in one output tile column) and, block by block, gathers their input face-rows into
// staging, permutes them into an image laid out in output byte order, and writes the image to DRAM.
// Both data-movement RISC-Vs of a core run this source on disjoint atom ranges, each on its own NoC.
//
// Tile geometry: a 32x32 tile is 4 faces of 16x16 in the order (top-left, top-right, bottom-left,
// bottom-right); a face-row (16 elements, F bytes) is the largest unit contiguous in both tensors.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "g6_reshape_permute.hpp"

namespace {

constexpr uint32_t kTile = 32;
constexpr uint32_t kFace = 16;
constexpr uint32_t kNoRun = 0xFFFFFFFF;

FORCE_INLINE uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }
FORCE_INLINE uint32_t max_u32(uint32_t a, uint32_t b) { return a > b ? a : b; }

}  // namespace

void kernel_main() {
    constexpr uint32_t kElem = get_compile_time_arg_val(0);
    constexpr uint32_t kWIn = get_compile_time_arg_val(1);
    constexpr uint32_t kHIn = get_compile_time_arg_val(2);
    constexpr uint32_t kWtIn = get_compile_time_arg_val(3);
    constexpr uint32_t kHtIn = get_compile_time_arg_val(4);
    constexpr uint32_t kWOut = get_compile_time_arg_val(5);
    constexpr uint32_t kHOut = get_compile_time_arg_val(6);
    constexpr uint32_t kBOut = get_compile_time_arg_val(7);
    constexpr uint32_t kWtOut = get_compile_time_arg_val(8);
    constexpr uint32_t kHtOut = get_compile_time_arg_val(9);
    constexpr bool kFaceRowAligned = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t kAtomRows = get_compile_time_arg_val(11);
    constexpr uint32_t kBlockAtoms = get_compile_time_arg_val(12);
    constexpr uint32_t kDramAlign = get_compile_time_arg_val(13);
    constexpr uint32_t kStageDepth = get_compile_time_arg_val(14);
    constexpr uint32_t kImageDepth = get_compile_time_arg_val(15);
    constexpr uint32_t kStageSlotBytes = get_compile_time_arg_val(16);
    constexpr uint32_t kImageSlotBytes = get_compile_time_arg_val(17);
    constexpr uint32_t kMaxCopies = get_compile_time_arg_val(18);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(19);
    constexpr uint32_t cb_image = get_compile_time_arg_val(20);
    constexpr uint32_t cb_plan = get_compile_time_arg_val(21);
    constexpr bool kStoreDirect = get_compile_time_arg_val(22) != 0;
    constexpr auto in_args = TensorAccessorArgs<23>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    constexpr uint32_t kFaceRowBytes = kFace * kElem;
    constexpr uint32_t kFaceBytes = kFace * kFaceRowBytes;
    constexpr uint32_t kTileBytes = 4 * kFaceBytes;
    constexpr uint32_t kAtomsPerImage = (kHOut + kAtomRows - 1) / kAtomRows;
    constexpr uint32_t kAtomsPerCol = kBOut * kAtomsPerImage;
    // Consecutive output rows are kWOut apart in the stream; precomputed so the walk only adds.
    constexpr uint32_t kRowStepRows = kWOut / kWIn;
    constexpr uint32_t kRowStepCols = kWOut % kWIn;
    constexpr uint32_t kRowStepImages = kRowStepRows / kHIn;
    constexpr uint32_t kRowStepH = kRowStepRows % kHIn;
    static_assert((kDramAlign & (kDramAlign - 1)) == 0, "DRAM alignment must be a power of two");
    static_assert(kFaceRowBytes <= 0xFF, "plan entries hold a copy length in 8 bits");

    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t atom_begin = get_arg_val<uint32_t>(0);
    const uint32_t atom_end = atom_begin + get_arg_val<uint32_t>(1);

    const auto in = TensorAccessor(in_args, in_addr, kTileBytes);
    const auto out = TensorAccessor(out_args, out_addr, kTileBytes);
    const uint8_t noc = noc_index;

    const uint32_t stage_cb_base = get_write_ptr(cb_stage);
    const uint32_t image_cb_base = get_write_ptr(cb_image);

    if constexpr (kStoreDirect) {
        static_assert(
            !kStoreDirect || (kHOut == 1 && kFaceRowAligned && kHtOut == 1),
            "lean path: one row per output tile, face-rows");
        // Consecutive output rows (atoms of one tile column) are consecutive input rows: runs coalesce.
        constexpr bool kCoalesce = kWOut == kWIn;
        uint32_t stage_slot = 0;
        for (uint32_t a0 = atom_begin; a0 < atom_end; a0 += kBlockAtoms) {
            const uint32_t a1 = min_u32(a0 + kBlockAtoms, atom_end);
            const uint32_t stage_base = stage_cb_base + stage_slot * kStageSlotBytes;

            // Write list (in the plan CB): {staging src, dst NoC addr lo, hi} per output face-row, built between the
            // read issue and the read barrier so the write-side address math hides in the DRAM read latency.
            uint32_t* const wl_base = reinterpret_cast<uint32_t*>(get_write_ptr(cb_plan));
            uint32_t* wl = wl_base;
            // One pass over the block's read runs: issue == true reads them, false lists their face-row writes.
            auto runs = [&](const bool issue) __attribute__((always_inline)) {
                uint32_t cursor = stage_base;
                uint32_t a = a0;
                while (a < a1) {
                    const uint32_t tc = a / kBOut;
                    const uint32_t b0 = a - tc * kBOut;
                    const uint32_t seg_end = min_u32(a1, (tc + 1) * kBOut);
                    const uint32_t nrows = seg_end - a;
                    const uint32_t halves = kWOut - kTile * tc > kFace ? 2 : 1;
                    for (uint32_t half = 0; half < halves; ++half) {
                        const uint32_t stream = b0 * kWOut + kTile * tc + kFace * half;
                        const uint32_t r = stream / kWIn;
                        uint32_t c = stream - r * kWIn;
                        uint32_t in_b = r / kHIn;
                        uint32_t in_h = r - in_b * kHIn;
                        uint32_t run_page = kNoRun;
                        uint32_t run_off = 0;
                        uint32_t run_rows = 0;
                        uint32_t run_stage = 0;
                        uint32_t run_b = 0;
                        auto flush = [&]() __attribute__((always_inline)) {
                            if (issue) {
                                noc_async_read_one_packet(
                                    in.get_noc_addr(run_page, run_off, noc), run_stage, run_rows * kFaceRowBytes, noc);
                            } else {
                                for (uint32_t k = 0; k < run_rows; ++k) {
                                    const uint64_t dst =
                                        out.get_noc_addr((run_b + k) * kWtOut + tc, half * kFaceBytes, noc);
                                    wl[0] = run_stage + k * kFaceRowBytes;
                                    wl[1] = static_cast<uint32_t>(dst);
                                    wl[2] = static_cast<uint32_t>(dst >> 32);
                                    wl += 3;
                                }
                            }
                            cursor = run_stage + run_rows * kFaceRowBytes;
                        };
                        for (uint32_t i = 0; i < nrows; ++i) {
                            const uint32_t page = (in_b * kHtIn + in_h / kTile) * kWtIn + c / kTile;
                            const uint32_t fr_off = (2 * ((in_h % kTile) / kFace) + (c % kTile) / kFace) * kFaceBytes +
                                                    (in_h % kFace) * kFaceRowBytes;
                            if (kCoalesce && page == run_page && fr_off == run_off + run_rows * kFaceRowBytes) {
                                ++run_rows;
                            } else {
                                if (run_rows > 0) {
                                    flush();
                                }
                                run_stage = cursor + (((fr_off & (kDramAlign - 1)) - cursor) & (kDramAlign - 1));
                                run_page = page;
                                run_off = fr_off;
                                run_rows = 1;
                                run_b = b0 + i;
                            }
                            c += kRowStepCols;
                            uint32_t carry = 0;
                            if (c >= kWIn) {
                                c -= kWIn;
                                carry = 1;
                            }
                            in_h += kRowStepH + carry;
                            in_b += kRowStepImages;
                            if (in_h >= kHIn) {
                                in_h -= kHIn;
                                ++in_b;
                            }
                        }
                        flush();
                        ASSERT(cursor <= stage_base + kStageSlotBytes);
                    }
                    a = seg_end;
                }
            };

            {
                DeviceZoneScopedN("g6.gather");
                if (a0 != atom_begin) {
                    // The previous block's writes may still source this staging slot.
                    noc_async_writes_flushed(noc);
                }
                runs(true);
                runs(false);
            }
            {
                DeviceZoneScopedN("g6.await");
                noc_async_read_barrier(noc);
            }
            {
                DeviceZoneScopedN("g6.store");
                for (const uint32_t* e = wl_base; e != wl; e += 3) {
                    noc_async_write_one_packet(e[0], (static_cast<uint64_t>(e[2]) << 32) | e[1], kFaceRowBytes, noc);
                }
            }
            stage_slot = stage_slot + 1 == kStageDepth ? 0 : stage_slot + 1;
        }
        {
            DeviceZoneScopedN("g6.drain");
            noc_async_write_barrier(noc);
        }
        return;
    }

    // Plan page: kMaxCopies copy entries {src L1 addr, (image offset << 8) | bytes}, then the tile table
    // {output page, image offset, lo | hi << 8 | halves << 16, unused} with one entry per output tile.
    volatile tt_l1_ptr uint32_t* copies = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_plan));
    volatile tt_l1_ptr uint32_t* tiles = copies + 2 * kMaxCopies;
    // lean_pre_all: the image path's store list follows the tile table (<= 4 write runs per output tile).
    uint32_t* const sl_base =
        reinterpret_cast<uint32_t*>(get_write_ptr(cb_plan) + (2 * kMaxCopies + 4 * kBlockAtoms) * 4);

    uint32_t stage_slot = 0;
    uint32_t image_slot = 0;
    for (uint32_t a0 = atom_begin; a0 < atom_end; a0 += kBlockAtoms) {
        const uint32_t a1 = min_u32(a0 + kBlockAtoms, atom_end);
        const uint32_t stage_base = stage_cb_base + stage_slot * kStageSlotBytes;
        const uint32_t image_base = image_cb_base + image_slot * kImageSlotBytes;
        uint32_t num_copies = 0;
        uint32_t num_tiles = 0;
        uint32_t* sl = sl_base;

        {
            DeviceZoneScopedN("g6.gather");
            // Open read run: [run_src_off, run_src_end) of input page run_page, landing at run_stage.
            uint32_t run_page = kNoRun;
            uint32_t run_src_off = 0;
            uint32_t run_src_end = 0;
            uint32_t run_stage = stage_base;
            uint64_t run_noc_addr = 0;

            auto close_run = [&]() {
                if (run_page != kNoRun) {
                    noc_async_read(run_noc_addr, run_stage, run_src_end - run_src_off, noc);
                }
            };

            // Staging address of the input face-row at (page, fr_off): reused when the open run holds it,
            // appended when it is the next face-row of the run's page, else read by a new run whose
            // staging address matches the source address modulo the DRAM read alignment.
            auto stage_face_row = [&](uint32_t page, uint32_t fr_off) -> uint32_t {
                if (page == run_page && fr_off >= run_src_off && fr_off < run_src_end) {
                    return run_stage + (fr_off - run_src_off);
                }
                if (page == run_page && fr_off == run_src_end) {
                    run_src_end += kFaceRowBytes;
                    return run_stage + (fr_off - run_src_off);
                }
                const uint32_t cursor = run_page == kNoRun ? stage_base : run_stage + (run_src_end - run_src_off);
                close_run();
                run_noc_addr = in.get_noc_addr(page, fr_off, noc);
                const uint32_t phase = static_cast<uint32_t>(run_noc_addr) & (kDramAlign - 1);
                run_stage = cursor + ((phase - cursor) & (kDramAlign - 1));
                run_page = page;
                run_src_off = fr_off;
                run_src_end = fr_off + kFaceRowBytes;
                ASSERT(run_stage + kFaceRowBytes <= stage_base + kStageSlotBytes);
                return run_stage;
            };

            uint32_t image_off = 0;
            uint32_t a = a0;
            while (a < a1) {
                // Column segment: the block's atoms in output tile column tc are consecutive output rows.
                const uint32_t tc = a / kAtomsPerCol;
                const uint32_t q_first = a % kAtomsPerCol;
                const uint32_t seg_end = min_u32(a1, (tc + 1) * kAtomsPerCol);
                const uint32_t q_last = q_first + (seg_end - a) - 1;
                const uint32_t row_begin = (q_first / kAtomsPerImage) * kHOut + (q_first % kAtomsPerImage) * kAtomRows;
                const uint32_t row_end =
                    (q_last / kAtomsPerImage) * kHOut + min_u32(kHOut, (q_last % kAtomsPerImage + 1) * kAtomRows);
                const uint32_t col_left = kWOut - kTile * tc;
                const uint32_t halves = col_left > kFace ? 2 : 1;

                // Output tile entries of the segment, with each tile's compacted image region.
                const uint32_t seg_first_tile = num_tiles;
                {
                    uint32_t b = row_begin / kHOut;
                    uint32_t h = row_begin % kHOut;
                    uint32_t rows_left = row_end - row_begin;
                    while (rows_left > 0) {
                        const uint32_t lo = h % kTile;
                        const uint32_t rows = min_u32(min_u32(kTile - lo, kHOut - h), rows_left);
                        const uint32_t hi = lo + rows;
                        // An atom never straddles an output tile, so a block has at most kBlockAtoms tiles.
                        ASSERT(num_tiles < kBlockAtoms);
                        volatile tt_l1_ptr uint32_t* t = tiles + 4 * num_tiles;
                        t[0] = (b * kHtOut + h / kTile) * kWtOut + tc;
                        t[1] = image_off;
                        t[2] = lo | (hi << 8) | (halves << 16);
                        ++num_tiles;
                        image_off += halves * rows * kFaceRowBytes;
                        h += rows;
                        rows_left -= rows;
                        if (h == kHOut) {
                            h = 0;
                            ++b;
                        }
                    }
                }
                ASSERT(image_off <= kImageSlotBytes);

                // Face-column walk: all rows of half 0, then all rows of half 1, so consecutive output
                // face-rows of one face column pull consecutive face-rows of one input face column.
                for (uint32_t half = 0; half < halves; ++half) {
                    const uint32_t n = min_u32(kFace, col_left - kFace * half);
                    const uint64_t stream = static_cast<uint64_t>(row_begin) * kWOut + kTile * tc + kFace * half;
                    const uint32_t r = static_cast<uint32_t>(stream / kWIn);
                    uint32_t c = static_cast<uint32_t>(stream % kWIn);
                    uint32_t in_b = r / kHIn;
                    uint32_t in_h = r % kHIn;

                    for (uint32_t ti = seg_first_tile; ti < num_tiles; ++ti) {
                        volatile tt_l1_ptr uint32_t* t = tiles + 4 * ti;
                        const uint32_t lo = t[2] & 0xFF;
                        const uint32_t hi = (t[2] >> 8) & 0xFF;
                        const uint32_t n0 = lo < kFace ? min_u32(hi, kFace) - lo : 0;
                        const uint32_t n1 = hi > kFace ? hi - max_u32(lo, kFace) : 0;
                        const uint32_t top_dst = t[1] + half * n0 * kFaceRowBytes;
                        const uint32_t bot_dst = t[1] + (halves * n0 + half * n1) * kFaceRowBytes;
                        const uint32_t bot_lo = max_u32(lo, kFace);

                        for (uint32_t rho = lo; rho < hi; ++rho) {
                            const uint32_t dst = rho < kFace ? top_dst + (rho - lo) * kFaceRowBytes
                                                             : bot_dst + (rho - bot_lo) * kFaceRowBytes;

                            // Pieces of this output face-row: each stays inside one input face-row and
                            // inside the logical input row, so input padding is never copied.
                            uint32_t pb = in_b;
                            uint32_t ph = in_h;
                            uint32_t pc = c;
                            uint32_t k = 0;
                            while (true) {
                                const uint32_t c_in_fr = pc % kFace;
                                const uint32_t len =
                                    kFaceRowAligned ? kFace : min_u32(min_u32(n - k, kFace - c_in_fr), kWIn - pc);
                                const uint32_t page = (pb * kHtIn + ph / kTile) * kWtIn + pc / kTile;
                                const uint32_t fr_off =
                                    (2 * ((ph % kTile) / kFace) + (pc % kTile) / kFace) * kFaceBytes +
                                    (ph % kFace) * kFaceRowBytes;
                                const uint32_t src = stage_face_row(page, fr_off) + c_in_fr * kElem;
                                ASSERT(num_copies < kMaxCopies);
                                copies[2 * num_copies] = src;
                                copies[2 * num_copies + 1] = ((dst + k * kElem) << 8) | (len * kElem);
                                ++num_copies;
                                if (kFaceRowAligned) {
                                    break;
                                }
                                k += len;
                                if (k == n) {
                                    break;
                                }
                                pc += len;
                                if (pc == kWIn) {
                                    pc = 0;
                                    if (++ph == kHIn) {
                                        ph = 0;
                                        ++pb;
                                    }
                                }
                            }

                            c += kRowStepCols;
                            uint32_t carry = 0;
                            if (c >= kWIn) {
                                c -= kWIn;
                                carry = 1;
                            }
                            in_h += kRowStepH + carry;
                            in_b += kRowStepImages;
                            if (in_h >= kHIn) {
                                in_h -= kHIn;
                                ++in_b;
                            }
                        }
                    }
                }
                a = seg_end;
            }
            close_run();
            {
                // Store list, built in the read shadow: {image src, dst NoC addr lo, hi, bytes} per merged write run.
                // Per output tile, regions f0 f1 f2 f3 in DRAM byte order (f1 / f3 only with two halves),
                // merged into one write whenever byte-contiguous in the page.
                for (uint32_t ti = 0; ti < num_tiles; ++ti) {
                    volatile tt_l1_ptr uint32_t* t = tiles + 4 * ti;
                    const uint32_t page = t[0];
                    uint32_t img = image_base + t[1];
                    const uint32_t lo = t[2] & 0xFF;
                    const uint32_t hi = (t[2] >> 8) & 0xFF;
                    const uint32_t halves = t[2] >> 16;
                    const uint32_t n0 = lo < kFace ? min_u32(hi, kFace) - lo : 0;
                    const uint32_t n1 = hi > kFace ? hi - max_u32(lo, kFace) : 0;
                    const uint32_t bot_row = max_u32(lo, kFace) - kFace;

                    uint32_t run_off = 0;
                    uint32_t run_len = 0;
                    uint32_t run_img = img;
                    for (uint32_t face = 0; face < 4; ++face) {
                        const uint32_t face_half = face & 1;
                        if (face_half >= halves) {
                            continue;
                        }
                        const uint32_t rows = face < 2 ? n0 : n1;
                        if (rows == 0) {
                            continue;
                        }
                        const uint32_t off = face * kFaceBytes + (face < 2 ? lo : bot_row) * kFaceRowBytes;
                        const uint32_t len = rows * kFaceRowBytes;
                        if (run_len > 0 && off == run_off + run_len) {
                            run_len += len;
                        } else {
                            if (run_len > 0) {
                                {
                                    const uint64_t dst = out.get_noc_addr(page, run_off, noc);
                                    sl[0] = run_img;
                                    sl[1] = static_cast<uint32_t>(dst);
                                    sl[2] = static_cast<uint32_t>(dst >> 32);
                                    sl[3] = run_len;
                                    sl += 4;
                                }
                            }
                            run_off = off;
                            run_len = len;
                            run_img = img;
                        }
                        img += len;
                    }
                    {
                        const uint64_t dst = out.get_noc_addr(page, run_off, noc);
                        sl[0] = run_img;
                        sl[1] = static_cast<uint32_t>(dst);
                        sl[2] = static_cast<uint32_t>(dst >> 32);
                        sl[3] = run_len;
                        sl += 4;
                    }
                }
            }
        }

        {
            DeviceZoneScopedN("g6.await");
            noc_async_read_barrier(noc);
        }

        {
            DeviceZoneScopedN("g6.permute");
            g6_reshape::permute_block<kElem, kFaceRowAligned ? kFaceRowBytes / 4 : 0>(
                copies, num_copies, image_base, noc);
        }

        {
            DeviceZoneScopedN("g6.store");
            for (const uint32_t* e = sl_base; e != sl; e += 4) {
                noc_async_write(e[0], (static_cast<uint64_t>(e[2]) << 32) | e[1], e[3], noc);
            }
        }

        stage_slot = stage_slot + 1 == kStageDepth ? 0 : stage_slot + 1;
        image_slot = image_slot + 1 == kImageDepth ? 0 : image_slot + 1;
    }

    {
        DeviceZoneScopedN("g6.drain");
        noc_async_write_barrier(noc);
    }
}
