// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// BENCH VARIANT pf1 (planfree_tile_mover): mover_pf.cpp with the walk as ONE non-inlined body run twice with a
// runtime pass flag (instead of two template instantiations), so pass 1 re-executes code pass 0 already pulled
// into the instruction cache.
//
// Plan-free g6_reshape tile mover (tile_face_row and tile_element_run).
//
// Same block skeleton and the same transfers as the op's mover (gather -> await -> place -> store), but no plan
// CB: the block walk is a pure function of the atom range, and the staging placement of every read run is a pure
// function of the walk's (page, face-row offset) sequence (a run lands at the next staging address congruent to
// its source offset mod A: the input buffer base and its tile page pitch are A-aligned, so the source phase is
// fr_off mod A, no NoC address needed). So the walk runs twice with one template body:
//   pass 0 (g6.gather): open/extend read runs and issue them;
//   pass 1 (g6.permute, after the barrier): replay the same run state machine to get each piece's staging
//     address, then copy it into the image (image stores), or write it straight from staging to its output
//     face-row (direct stores, CT kStoreDirect: tile_face_row with H_out == 1, where every output face-row is its
//     own write in the image path too, so no write merge is lost; no CPU copy, cache invalidate or fence).
// The image store (g6.store) walks the block's output tiles in closed form (no tile table).
// Zones: g6.gather, g6.await, g6.permute (image: copies; direct: the writes), g6.store (image only), g6.drain.
// No raw LLK: dataflow API only.

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

template <int N>
struct PassT {
    static constexpr int value = N;
};

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
    constexpr uint32_t cb_stage = get_compile_time_arg_val(19);
    constexpr uint32_t cb_image = get_compile_time_arg_val(20);
    constexpr bool kStoreDirect = get_compile_time_arg_val(22) != 0;
    constexpr auto in_args = TensorAccessorArgs<23>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    constexpr uint32_t kFaceRowBytes = kFace * kElem;
    constexpr uint32_t kFaceBytes = kFace * kFaceRowBytes;
    constexpr uint32_t kTileBytes = 4 * kFaceBytes;
    constexpr uint32_t kAtomsPerImage = (kHOut + kAtomRows - 1) / kAtomRows;
    constexpr uint32_t kAtomsPerCol = kBOut * kAtomsPerImage;
    constexpr uint32_t kRowStepRows = kWOut / kWIn;
    constexpr uint32_t kRowStepCols = kWOut % kWIn;
    constexpr uint32_t kRowStepImages = kRowStepRows / kHIn;
    constexpr uint32_t kRowStepH = kRowStepRows % kHIn;
    static_assert((kDramAlign & (kDramAlign - 1)) == 0, "DRAM alignment must be a power of two");
    static_assert(kTileBytes % kDramAlign == 0, "tile pages keep the source phase = face-row offset mod A");
    static_assert(!kStoreDirect || kFaceRowAligned, "direct stores need whole face-rows (16-B aligned on both ends)");

    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t atom_begin = get_arg_val<uint32_t>(0);
    const uint32_t atom_end = atom_begin + get_arg_val<uint32_t>(1);

    const auto in = TensorAccessor(in_args, in_addr, kTileBytes);
    const auto out = TensorAccessor(out_args, out_addr, kTileBytes);
    const uint8_t noc = noc_index;

    const uint32_t stage_cb_base = get_write_ptr(cb_stage);
    const uint32_t image_cb_base = get_write_ptr(cb_image);

    uint32_t stage_slot = 0;
    uint32_t image_slot = 0;
    for (uint32_t a0 = atom_begin; a0 < atom_end; a0 += kBlockAtoms) {
        const uint32_t a1 = min_u32(a0 + kBlockAtoms, atom_end);
        const uint32_t stage_base = stage_cb_base + stage_slot * kStageSlotBytes;
        const uint32_t image_base = image_cb_base + image_slot * kImageSlotBytes;

        // The block's walk: column segments, face-column order (all rows of half 0, then half 1), pieces.
        // Pass 0 issues the read runs; pass 1 replays the identical run placement and places each piece.
        uint32_t last_byte = image_base;
        auto walk = [&](const bool kIssue) __attribute__((noinline)) {
            uint32_t run_page = kNoRun;
            uint32_t run_src_off = 0;
            uint32_t run_src_end = 0;
            uint32_t run_stage = stage_base;

            auto stage_face_row = [&](uint32_t page, uint32_t fr_off) __attribute__((always_inline)) -> uint32_t {
                if (page == run_page) {
                    if (fr_off >= run_src_off && fr_off < run_src_end) {
                        return run_stage + (fr_off - run_src_off);
                    }
                    if (fr_off == run_src_end) {
                        run_src_end += kFaceRowBytes;
                        return run_stage + (fr_off - run_src_off);
                    }
                }
                const uint32_t cursor = run_page == kNoRun ? stage_base : run_stage + (run_src_end - run_src_off);
                if (kIssue) {
                    if (run_page != kNoRun) {
                        noc_async_read(
                            in.get_noc_addr(run_page, run_src_off, noc), run_stage, run_src_end - run_src_off, noc);
                    }
                }
                const uint32_t phase = fr_off & (kDramAlign - 1);
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
                const uint32_t tc = a / kAtomsPerCol;
                const uint32_t q_first = a % kAtomsPerCol;
                const uint32_t seg_end = min_u32(a1, (tc + 1) * kAtomsPerCol);
                const uint32_t q_last = q_first + (seg_end - a) - 1;
                const uint32_t row_begin = (q_first / kAtomsPerImage) * kHOut + (q_first % kAtomsPerImage) * kAtomRows;
                const uint32_t row_end =
                    (q_last / kAtomsPerImage) * kHOut + min_u32(kHOut, (q_last % kAtomsPerImage + 1) * kAtomRows);
                const uint32_t col_left = kWOut - kTile * tc;
                const uint32_t halves = col_left > kFace ? 2 : 1;
                const uint32_t b_begin = row_begin / kHOut;
                const uint32_t h_begin = row_begin % kHOut;

                for (uint32_t half = 0; half < halves; ++half) {
                    const uint32_t n = min_u32(kFace, col_left - kFace * half);
                    const uint64_t stream = static_cast<uint64_t>(row_begin) * kWOut + kTile * tc + kFace * half;
                    const uint32_t r = static_cast<uint32_t>(stream / kWIn);
                    uint32_t c = static_cast<uint32_t>(stream % kWIn);
                    uint32_t in_b = r / kHIn;
                    uint32_t in_h = r % kHIn;

                    // Output tiles of the segment, closed form (same order and image layout as the store).
                    uint32_t b = b_begin;
                    uint32_t h = h_begin;
                    uint32_t rows_left = row_end - row_begin;
                    uint32_t tile_img = image_off;
                    while (rows_left > 0) {
                        const uint32_t lo = h % kTile;
                        const uint32_t rows = min_u32(min_u32(kTile - lo, kHOut - h), rows_left);
                        const uint32_t hi = lo + rows;
                        const uint32_t out_page = (b * kHtOut + h / kTile) * kWtOut + tc;
                        const uint32_t n0 = lo < kFace ? min_u32(hi, kFace) - lo : 0;
                        const uint32_t n1 = hi > kFace ? hi - max_u32(lo, kFace) : 0;
                        const uint32_t top_dst = tile_img + half * n0 * kFaceRowBytes;
                        const uint32_t bot_dst = tile_img + (halves * n0 + half * n1) * kFaceRowBytes;
                        const uint32_t bot_lo = max_u32(lo, kFace);

                        for (uint32_t rho = lo; rho < hi; ++rho) {
                            const uint32_t dst = rho < kFace ? top_dst + (rho - lo) * kFaceRowBytes
                                                             : bot_dst + (rho - bot_lo) * kFaceRowBytes;
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
                                if (!kIssue) {
                                    if constexpr (kStoreDirect) {
                                        const uint32_t face = (rho < kFace ? 0 : 2) + half;
                                        const uint32_t off = face * kFaceBytes + (rho % kFace) * kFaceRowBytes;
                                        noc_async_write(src, out.get_noc_addr(out_page, off, noc), kFaceRowBytes, noc);
                                    } else {
                                        const uint32_t to = image_base + dst + k * kElem;
                                        g6_reshape::copy_run<kElem, kFaceRowAligned ? kFaceRowBytes / 4 : 0>(
                                            src, to, len * kElem);
                                        last_byte = to + len * kElem - 1;
                                    }
                                }
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

                        tile_img += halves * rows * kFaceRowBytes;
                        h += rows;
                        rows_left -= rows;
                        if (h == kHOut) {
                            h = 0;
                            ++b;
                        }
                    }
                    if (half + 1 == halves) {
                        image_off = tile_img;
                    }
                }
                a = seg_end;
            }
            if (kIssue) {
                if (run_page != kNoRun) {
                    noc_async_read(
                        in.get_noc_addr(run_page, run_src_off, noc), run_stage, run_src_end - run_src_off, noc);
                }
            }
            ASSERT(image_off <= kImageSlotBytes);
        };

        {
            DeviceZoneScopedN("g6.gather");
            if constexpr (kStoreDirect) {
                // The previous block's direct writes may still source this staging slot.
                noc_async_writes_flushed(noc);
            }
            walk(true);
        }

        {
            DeviceZoneScopedN("g6.await");
            noc_async_read_barrier(noc);
        }

        if constexpr (kStoreDirect) {
            DeviceZoneScopedN("g6.permute");
            walk(false);
        } else {
            {
                DeviceZoneScopedN("g6.permute");
                g6_reshape::begin_image(noc);
                walk(false);
                g6_reshape::fence_image(last_byte);
            }
            {
                DeviceZoneScopedN("g6.store");
                uint32_t img = image_base;
                uint32_t a = a0;
                while (a < a1) {
                    const uint32_t tc = a / kAtomsPerCol;
                    const uint32_t q_first = a % kAtomsPerCol;
                    const uint32_t seg_end = min_u32(a1, (tc + 1) * kAtomsPerCol);
                    const uint32_t q_last = q_first + (seg_end - a) - 1;
                    const uint32_t row_begin =
                        (q_first / kAtomsPerImage) * kHOut + (q_first % kAtomsPerImage) * kAtomRows;
                    const uint32_t row_end =
                        (q_last / kAtomsPerImage) * kHOut + min_u32(kHOut, (q_last % kAtomsPerImage + 1) * kAtomRows);
                    const uint32_t halves = kWOut - kTile * tc > kFace ? 2 : 1;
                    uint32_t b = row_begin / kHOut;
                    uint32_t h = row_begin % kHOut;
                    uint32_t rows_left = row_end - row_begin;
                    while (rows_left > 0) {
                        const uint32_t lo = h % kTile;
                        const uint32_t rows = min_u32(min_u32(kTile - lo, kHOut - h), rows_left);
                        const uint32_t hi = lo + rows;
                        const uint32_t page = (b * kHtOut + h / kTile) * kWtOut + tc;
                        const uint32_t n0 = lo < kFace ? min_u32(hi, kFace) - lo : 0;
                        const uint32_t n1 = hi > kFace ? hi - max_u32(lo, kFace) : 0;
                        const uint32_t bot_row = max_u32(lo, kFace) - kFace;
                        uint32_t run_off = 0;
                        uint32_t run_len = 0;
                        uint32_t run_img = img;
                        for (uint32_t face = 0; face < 4; ++face) {
                            if ((face & 1) >= halves) {
                                continue;
                            }
                            const uint32_t frows = face < 2 ? n0 : n1;
                            if (frows == 0) {
                                continue;
                            }
                            const uint32_t off = face * kFaceBytes + (face < 2 ? lo : bot_row) * kFaceRowBytes;
                            const uint32_t len = frows * kFaceRowBytes;
                            if (run_len > 0 && off == run_off + run_len) {
                                run_len += len;
                            } else {
                                if (run_len > 0) {
                                    noc_async_write(run_img, out.get_noc_addr(page, run_off, noc), run_len, noc);
                                }
                                run_off = off;
                                run_len = len;
                                run_img = img;
                            }
                            img += len;
                        }
                        noc_async_write(run_img, out.get_noc_addr(page, run_off, noc), run_len, noc);
                        h += rows;
                        rows_left -= rows;
                        if (h == kHOut) {
                            h = 0;
                            ++b;
                        }
                    }
                    a = seg_end;
                }
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
