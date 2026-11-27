// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One g6_reshape mover for the rm_row_gather regime (ROW_MAJOR, page = one logical row).
//
// Same skeleton as the tile mover (gather -> await -> permute -> store, one block at a time), with the
// row-major geometry: an atom is one A-byte chunk of an output page slot (A = DRAM alignment, the slot is
// align(W_out*e, A) bytes). Two atom folds (CT kLinearFold, host knob RM_FOLD):
//  - linear (default): atoms in stream order (page, chunk), so a block is ONE contiguous range of the input
//    stream (each input byte read once, no stride amplification on a hot input bank); the permute lays the
//    image out BANK-GROUPED (pages p0+g, p0+g+NB, ... consecutively), so the block still stores as one write
//    per (block, bank) — at most NB writes.
//  - bank-affine: ordered by (bank, page / NB, chunk); a mover's consecutive atoms are consecutive bytes of
//    one DRAM bank (fewest writes), but its input elements are NB pages apart in the stream. Reads are A-aligned
//    covering runs inside one input page (gaps up to kReadGap bytes are read
// through), landed at A-aligned staging addresses, so every NoC transfer is DRAM-aligned on both ends.
//
// Why a second source rather than a branch of g6_reshape_mover.cpp: the gather walk and the store geometry
// share nothing with the tile path (rows vs faces, bank slots vs tile pages) and the CT-arg layouts are
// disjoint; the shared block operation (permute) lives in g6_reshape_permute.hpp and is used by both.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "g6_reshape_permute.hpp"

namespace {

constexpr uint32_t kNoRun = 0xFFFFFFFF;

FORCE_INLINE uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

}  // namespace

void kernel_main() {
    constexpr uint32_t kElem = get_compile_time_arg_val(0);
    constexpr uint32_t kWIn = get_compile_time_arg_val(1);
    constexpr uint32_t kWOut = get_compile_time_arg_val(2);
    constexpr uint32_t kROut = get_compile_time_arg_val(3);
    constexpr uint32_t kAtomsPerPage = get_compile_time_arg_val(4);
    constexpr uint32_t kBlockAtoms = get_compile_time_arg_val(5);
    constexpr uint32_t kAlign = get_compile_time_arg_val(6);
    constexpr uint32_t kReadGap = get_compile_time_arg_val(7);
    constexpr uint32_t kStageDepth = get_compile_time_arg_val(8);
    constexpr uint32_t kImageDepth = get_compile_time_arg_val(9);
    constexpr uint32_t kStageSlotBytes = get_compile_time_arg_val(10);
    constexpr uint32_t kImageSlotBytes = get_compile_time_arg_val(11);
    constexpr uint32_t kMaxCopies = get_compile_time_arg_val(12);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(13);
    constexpr uint32_t cb_image = get_compile_time_arg_val(14);
    constexpr uint32_t cb_plan = get_compile_time_arg_val(15);
    constexpr bool kLinearFold = get_compile_time_arg_val(16) != 0;
    constexpr auto in_args = TensorAccessorArgs<17>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    constexpr uint32_t kNumBanks = NUM_DRAM_BANKS;
    constexpr uint32_t kAtomElems = kAlign / kElem;
    constexpr uint32_t kInPageBytes = kWIn * kElem;
    constexpr uint32_t kOutPageBytes = kWOut * kElem;
    static_assert((kAlign & (kAlign - 1)) == 0, "DRAM alignment must be a power of two");
    static_assert(kAlign <= 0xFF, "plan entries hold a copy length in 8 bits");
    static_assert(kElem == 2 || kElem == 4, "rm_row_gather moves 2- or 4-byte elements");

    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t atom_begin = get_arg_val<uint32_t>(0);
    const uint32_t atom_count = get_arg_val<uint32_t>(1);
    const uint8_t noc = noc_index;

    // Interleaved accessors align the page size to the DRAM alignment themselves (the slot pitch).
    const auto in = TensorAccessor(in_args, in_addr, kInPageBytes);
    const auto out = TensorAccessor(out_args, out_addr, kOutPageBytes);

    // The CBs carry kAlign bytes of slack so the slots start DRAM-aligned whatever the CB base.
    const uint32_t stage_cb_base = (get_write_ptr(cb_stage) + kAlign - 1) & ~(kAlign - 1);
    const uint32_t image_cb_base = (get_write_ptr(cb_image) + kAlign - 1) & ~(kAlign - 1);
    volatile tt_l1_ptr uint32_t* copies = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_plan));

    // Fold cursor: output page `page`, chunk `chunk` of its slot; `bank` is the bank-affine fold's outer index
    // (unused by the linear fold).
    uint32_t bank = 0;
    uint32_t page = 0;
    uint32_t chunk = 0;
    if constexpr (kLinearFold) {
        page = atom_begin / kAtomsPerPage;
        chunk = atom_begin % kAtomsPerPage;
    } else if (atom_count > 0) {
        uint32_t rem = atom_begin;
        while (true) {
            const uint32_t bank_atoms = (bank < kROut ? (kROut - bank + kNumBanks - 1) / kNumBanks : 0) * kAtomsPerPage;
            if (rem < bank_atoms) {
                break;
            }
            rem -= bank_atoms;
            ++bank;
        }
        page = bank + kNumBanks * (rem / kAtomsPerPage);
        chunk = rem % kAtomsPerPage;
    }
    // Steps the cursor to the next atom in fold order; returns true when it left the bank.
    auto advance = [&](uint32_t& b, uint32_t& p, uint32_t& k) -> bool {
        if (++k < kAtomsPerPage) {
            return false;
        }
        k = 0;
        if constexpr (kLinearFold) {
            ++p;
            return true;
        }
        p += kNumBanks;
        if (p < kROut) {
            return false;
        }
        ++b;
        p = b;
        return true;
    };

    uint32_t stage_slot = 0;
    uint32_t image_slot = 0;
    const uint32_t atom_end = atom_begin + atom_count;
    for (uint32_t a0 = atom_begin; a0 < atom_end; a0 += kBlockAtoms) {
        const uint32_t n_atoms = min_u32(kBlockAtoms, atom_end - a0);
        const uint32_t stage_base = stage_cb_base + stage_slot * kStageSlotBytes;
        const uint32_t image_base = image_cb_base + image_slot * kImageSlotBytes;
        const uint32_t blk_bank = bank;
        const uint32_t blk_page = page;
        const uint32_t blk_chunk = chunk;
        uint32_t num_copies = 0;
        // Linear fold: the block's pages blk_page + j land in bank group g = j mod NB; the image holds group 0,
        // then group 1, ... (each group = consecutive bytes of one bank). group_off[g] is the group's image
        // offset, advanced as its atoms are placed; group_bytes[g] its store length.
        constexpr uint32_t kGroups = kLinearFold ? kNumBanks : 1;
        uint32_t group_off[kGroups];
        uint32_t group_bytes[kGroups];
        uint32_t num_groups = 1;
        if constexpr (kLinearFold) {
            const uint32_t last = blk_chunk + n_atoms - 1;  // chunk index of the block's last atom, from blk_page
            const uint32_t num_pages = last / kAtomsPerPage + 1;
            const uint32_t last_page_atoms = last % kAtomsPerPage + 1;
            num_groups = min_u32(kNumBanks, num_pages);
            uint32_t off = 0;
            for (uint32_t g = 0; g < num_groups; ++g) {
                uint32_t atoms = ((num_pages - g + kNumBanks - 1) / kNumBanks) * kAtomsPerPage;
                if (g == 0) {
                    atoms -= blk_chunk;
                }
                if (g == (num_pages - 1) % kNumBanks) {
                    atoms -= kAtomsPerPage - last_page_atoms;
                }
                group_off[g] = off;
                group_bytes[g] = atoms * kAlign;
                off += atoms * kAlign;
            }
        }

        // Linear fold: the block is one contiguous stream range, so its covering reads (one A-aligned run per
        // input row it crosses, staged back to back) are known before the walk — issue them first and let the
        // walk below overlap their DRAM latency. The walk lays staging out identically (runs only extend
        // contiguously in stream order) and only checks it.
        uint32_t issued_stage_end = stage_base;
        if constexpr (kLinearFold) {
            DeviceZoneScopedN("g6.issue");
            const uint32_t last = blk_chunk + n_atoms - 1;
            const uint32_t s0 = blk_page * kWOut + blk_chunk * kAtomElems;
            const uint32_t s1 =
                (blk_page + last / kAtomsPerPage) * kWOut + min_u32(kWOut, (last % kAtomsPerPage + 1) * kAtomElems);
            uint32_t r = s0 / kWIn;
            uint32_t c = s0 % kWIn;
            for (uint32_t pos = s0; pos < s1;) {
                const uint32_t len = min_u32(s1 - pos, kWIn - c);
                const uint32_t lo = (c * kElem) & ~(kAlign - 1);
                const uint32_t hi = ((c + len) * kElem + kAlign - 1) & ~(kAlign - 1);
                noc_async_read(in.get_noc_addr(r, lo, noc), issued_stage_end, hi - lo, noc);
                issued_stage_end += hi - lo;
                pos += len;
                c = 0;
                ++r;
            }
            ASSERT(issued_stage_end <= stage_base + kStageSlotBytes);
        }

        {
            DeviceZoneScopedN("g6.gather");
            // Open covering read run: bytes [run_lo, run_hi) of input page run_page (both A-aligned),
            // landing at run_stage.
            uint32_t run_page = kNoRun;
            uint32_t run_lo = 0;
            uint32_t run_hi = 0;
            uint32_t run_stage = stage_base;

            auto close_run = [&]() {
                if (!kLinearFold && run_page != kNoRun) {
                    noc_async_read(in.get_noc_addr(run_page, run_lo, noc), run_stage, run_hi - run_lo, noc);
                }
            };

            // Staging address of input bytes [off, off + bytes) of input page r.
            auto stage_piece = [&](uint32_t r, uint32_t off, uint32_t bytes) -> uint32_t {
                if (r == run_page && off >= run_lo && off <= run_hi + kReadGap) {
                    const uint32_t end = (off + bytes + kAlign - 1) & ~(kAlign - 1);
                    if (end > run_hi) {
                        run_hi = end;
                        ASSERT(run_stage + (run_hi - run_lo) <= stage_base + kStageSlotBytes);
                    }
                    return run_stage + (off - run_lo);
                }
                const uint32_t cursor = run_page == kNoRun ? stage_base : run_stage + (run_hi - run_lo);
                close_run();
                run_page = r;
                run_lo = off & ~(kAlign - 1);
                run_hi = (off + bytes + kAlign - 1) & ~(kAlign - 1);
                run_stage = cursor;
                ASSERT(run_stage + (run_hi - run_lo) <= stage_base + kStageSlotBytes);
                return run_stage + (off - run_lo);
            };

            uint32_t image_off = 0;
            uint32_t group = 0;
            for (uint32_t i = 0; i < n_atoms; ++i) {
                if constexpr (kLinearFold) {
                    image_off = group_off[group];
                    group_off[group] += kAlign;
                }
                // Valid elements of this chunk: [first, first + n) of the output stream.
                const uint32_t chunk_elem = chunk * kAtomElems;
                const uint32_t n = min_u32(kAtomElems, kWOut - chunk_elem);
                const uint32_t first = page * kWOut + chunk_elem;
                uint32_t r = first / kWIn;
                uint32_t c = first % kWIn;
                uint32_t k = 0;
                while (k < n) {
                    const uint32_t len = min_u32(n - k, kWIn - c);
                    const uint32_t src = stage_piece(r, c * kElem, len * kElem);
                    ASSERT(num_copies < kMaxCopies);
                    copies[2 * num_copies] = src;
                    copies[2 * num_copies + 1] = ((image_off + k * kElem) << 8) | (len * kElem);
                    ++num_copies;
                    k += len;
                    c = 0;
                    ++r;
                }
                if constexpr (kLinearFold) {
                    if (advance(bank, page, chunk)) {
                        group = group + 1 == kNumBanks ? 0 : group + 1;
                    }
                } else {
                    image_off += kAlign;
                    advance(bank, page, chunk);
                }
            }
            ASSERT(kLinearFold || image_off <= kImageSlotBytes);
            close_run();
            ASSERT(!kLinearFold || run_stage + (run_hi - run_lo) == issued_stage_end);
        }

        {
            DeviceZoneScopedN("g6.await");
            noc_async_read_barrier(noc);
        }

        {
            DeviceZoneScopedN("g6.permute");
            g6_reshape::permute_block<kElem, 0>(copies, num_copies, image_base, noc);
        }

        {
            DeviceZoneScopedN("g6.store");
            // One write per (block, bank): the block's atoms of one bank are consecutive bytes of that bank.
            if constexpr (kLinearFold) {
                // Group g starts at page blk_page + g (chunk blk_chunk for group 0) and its pages are NB apart,
                // i.e. consecutive slots of one bank.
                uint32_t img = image_base;
                for (uint32_t g = 0; g < num_groups; ++g) {
                    const uint32_t first_chunk = g == 0 ? blk_chunk : 0;
                    noc_async_write(
                        img, out.get_noc_addr(blk_page + g, first_chunk * kAlign, noc), group_bytes[g], noc);
                    img += group_bytes[g];
                }
                ASSERT(img - image_base == n_atoms * kAlign);
            } else {
                uint32_t b = blk_bank;
                uint32_t p = blk_page;
                uint32_t k = blk_chunk;
                uint32_t seg_page = p;
                uint32_t seg_chunk = k;
                uint32_t seg_img = image_base;
                uint32_t seg_len = 0;
                for (uint32_t i = 0; i < n_atoms; ++i) {
                    seg_len += kAlign;
                    const bool left_bank = advance(b, p, k);
                    if (left_bank || i + 1 == n_atoms) {
                        noc_async_write(seg_img, out.get_noc_addr(seg_page, seg_chunk * kAlign, noc), seg_len, noc);
                        seg_img += seg_len;
                        seg_len = 0;
                        seg_page = p;
                        seg_chunk = k;
                    }
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
