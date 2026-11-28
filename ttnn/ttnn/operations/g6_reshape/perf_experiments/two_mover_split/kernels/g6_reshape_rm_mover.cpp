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
//    per (block, bank) — at most NB writes. Plan-free (Refinement 3): the staged range is the input stream at
//    a uniform row pitch, so every copy source is closed-form; no gather walk and no plan in L1. The copies
//    run group by group and each group's write is issued as soon as its image bytes are fenced, so the
//    stores overlap the remaining copies.
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
    // Shared read (linear fold, one block per mover): one mover per core issues the covering reads of the whole
    // core's stream range into the shared staging CB (cb_stage is the same CB on both movers) and hands it to
    // its partner through the local semaphore kSemId; the readers alternate by core between the two NoCs.
    constexpr bool kSharedRead = get_compile_time_arg_val(17) != 0;
    constexpr uint32_t kSemId = get_compile_time_arg_val(18);
    // Bank split (shared read, one atom per output page): the core's movers split the core range's bank groups
    // (mover m takes groups m, m + kMovers, ...) instead of its stream halves, so each issues ~NB / kMovers
    // writes instead of ~NB. Same atom count per mover (NB is a multiple of kMovers).
    constexpr bool kBankSplit = get_compile_time_arg_val(19) != 0;
    constexpr uint32_t kMoverIdx = get_compile_time_arg_val(20);
    constexpr uint32_t kMovers = get_compile_time_arg_val(21);
    constexpr auto in_args = TensorAccessorArgs<22>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    constexpr uint32_t kNumBanks = NUM_DRAM_BANKS;
    constexpr uint32_t kAtomElems = kAlign / kElem;
    constexpr uint32_t kInPageBytes = kWIn * kElem;
    constexpr uint32_t kOutPageBytes = kWOut * kElem;
    static_assert((kAlign & (kAlign - 1)) == 0, "DRAM alignment must be a power of two");
    static_assert(kAlign <= 0xFF, "plan entries hold a copy length in 8 bits");
    static_assert(kElem == 2 || kElem == 4, "rm_row_gather moves 2- or 4-byte elements");
    static_assert(!kSharedRead || (kLinearFold && kStageDepth == 1), "the shared read stages one linear-fold block");
    static_assert(
        !kBankSplit || (kSharedRead && kAtomsPerPage == 1 && NUM_DRAM_BANKS % kMovers == 0),
        "the bank split needs the shared read, whole-page atoms and NB a multiple of the mover count");
    // Shared-read roles (RT arg 4).
    constexpr uint32_t kWaiter = 0;      // waits for the partner's reads
    constexpr uint32_t kReaderSolo = 1;  // reads; the partner owns no atoms
    constexpr uint32_t kReaderPair = 2;  // reads, then signals the partner

    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t atom_begin = get_arg_val<uint32_t>(0);
    const uint32_t atom_count = get_arg_val<uint32_t>(1);
    // Shared read: the core's atom range (both movers) and this mover's role.
    const uint32_t core_begin = kSharedRead ? get_arg_val<uint32_t>(2) : atom_begin;
    const uint32_t core_count = kSharedRead ? get_arg_val<uint32_t>(3) : atom_count;
    const uint32_t role = kSharedRead ? get_arg_val<uint32_t>(4) : kReaderSolo;
    const uint8_t noc = noc_index;

    // Interleaved accessors align the page size to the DRAM alignment themselves (the slot pitch).
    const auto in = TensorAccessor(in_args, in_addr, kInPageBytes);
    const auto out = TensorAccessor(out_args, out_addr, kOutPageBytes);

    // The CBs carry kAlign bytes of slack so the slots start DRAM-aligned whatever the CB base.
    const uint32_t stage_cb_base = (get_write_ptr(cb_stage) + kAlign - 1) & ~(kAlign - 1);
    const uint32_t image_cb_base = (get_write_ptr(cb_image) + kAlign - 1) & ~(kAlign - 1);
    // The move plan exists only for the bank fold (the linear fold has no plan CB).
    volatile tt_l1_ptr uint32_t* copies =
        kLinearFold ? nullptr : reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_plan));

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
    // Bank fold: steps the cursor to the next atom in fold order; returns true when it left the bank. (The
    // linear fold steps its cursor a whole block at a time.)
    auto advance = [&](uint32_t& b, uint32_t& p, uint32_t& k) -> bool {
        if (++k < kAtomsPerPage) {
            return false;
        }
        k = 0;
        p += kNumBanks;
        if (p < kROut) {
            return false;
        }
        ++b;
        p = b;
        return true;
    };

    // Linear fold: atoms [a, a + n) are the stream range [s0, s1) (chunks of consecutive output pages).
    auto stream_range = [&](uint32_t a, uint32_t n, uint32_t& s0, uint32_t& s1) {
        const uint32_t first_page = a / kAtomsPerPage;
        const uint32_t last = a % kAtomsPerPage + n - 1;  // chunk index of the last atom, from first_page
        s0 = first_page * kWOut + (a % kAtomsPerPage) * kAtomElems;
        s1 = (first_page + last / kAtomsPerPage) * kWOut + min_u32(kWOut, (last % kAtomsPerPage + 1) * kAtomElems);
    };
    // Covering reads of the stream range [s0, s1): one A-aligned run per input row it crosses, staged back to
    // back from `base`. Row r0 = s0 / W_in's run starts at byte lo0 = A-floor of s0's column, every later row's
    // at byte 0, and every run but the last ends at the row pitch, so stream element (r, c) is staged at
    // base - lo0 + (r - r0)·kInPitch + c·e (closed form: the copies need no plan). Returns the staging end.
    constexpr uint32_t kInPitch = (kInPageBytes + kAlign - 1) & ~(kAlign - 1);
    auto issue_range = [&](uint32_t s0, uint32_t s1, uint32_t base) -> uint32_t {
        DeviceZoneScopedN("g6.issue");
        const uint32_t r0 = s0 / kWIn;
        uint32_t r = r0;
        uint32_t c = s0 % kWIn;
        uint32_t end = base;
        for (uint32_t pos = s0; pos < s1;) {
            const uint32_t len = min_u32(s1 - pos, kWIn - c);
            const uint32_t lo = (c * kElem) & ~(kAlign - 1);
            const uint32_t hi = ((c + len) * kElem + kAlign - 1) & ~(kAlign - 1);
            ASSERT(r == r0 || end == base - ((s0 % kWIn) * kElem & ~(kAlign - 1)) + (r - r0) * kInPitch);
            noc_async_read(in.get_noc_addr(r, lo, noc), end, hi - lo, noc);
            end += hi - lo;
            pos += len;
            c = 0;
            ++r;
        }
        ASSERT(end <= base + kStageSlotBytes);
        return end;
    };

    // Shared read: the reader issues the whole core range up front (its latency overlaps both movers' setup).
    uint32_t core_s0 = 0;
    uint32_t core_stage_end = stage_cb_base + kStageSlotBytes;
    if constexpr (kSharedRead) {
        if (core_count > 0) {
            uint32_t core_s1;
            stream_range(core_begin, core_count, core_s0, core_s1);
            if (role != kWaiter) {
                core_stage_end = issue_range(core_s0, core_s1, stage_cb_base);
            }
        }
    }
    volatile tt_l1_ptr uint32_t* sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kSharedRead ? get_semaphore(kSemId) : 0);

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

        if constexpr (kLinearFold) {
            // The span this mover walks: the block (chunks blk_chunk .. last of pages span_page ..
            // span_page + num_pages - 1), or under the bank split the core's whole range, of which this mover takes
            // every kMovers-th bank group.
            const uint32_t span_page = kBankSplit ? core_begin : blk_page;
            const uint32_t span_atoms = kBankSplit ? core_count : n_atoms;
            constexpr uint32_t kGroup0 = kBankSplit ? kMoverIdx : 0;
            constexpr uint32_t kGroupStep = kBankSplit ? kMovers : 1;
            const uint32_t last = blk_chunk + span_atoms - 1;
            const uint32_t num_pages = last / kAtomsPerPage + 1;
            const uint32_t last_page_atoms = last % kAtomsPerPage + 1;
            // The staged range: the core's (shared read) or this block's, read now.
            uint32_t range_s0 = core_s0;
            uint32_t stage_end = core_stage_end;
            if constexpr (!kSharedRead) {
                uint32_t s1;
                stream_range(a0, n_atoms, range_s0, s1);
                stage_end = issue_range(range_s0, s1, stage_base);
            }
            const uint32_t r0 = range_s0 / kWIn;
            const uint32_t stage_row0 = stage_base - (((range_s0 % kWIn) * kElem) & ~(kAlign - 1));
            // Fast-path predicate: one atom per output page, and output pages never straddle an input row.
            constexpr bool kPageAtoms = kAtomsPerPage == 1 && kWIn % kWOut == 0;
            constexpr uint32_t kPagesPerRow = kWIn / kWOut;
            // Word count of a whole-page copy when it is word-aligned on both sides (0: byte-length copy).
            constexpr uint32_t kPageWords = kOutPageBytes % 4 == 0 ? kOutPageBytes / 4 : 0;

            // In the read latency's shadow: every group's output NoC address (group g starts at page span_page + g,
            // chunk blk_chunk for group 0), so the store below only issues.
            const uint32_t num_groups = min_u32(kNumBanks, num_pages);
            uint64_t group_dst[kNumBanks];
            for (uint32_t g = kGroup0; g < num_groups; g += kGroupStep) {
                group_dst[g] = out.get_noc_addr(span_page + g, (g == 0 ? blk_chunk : 0) * kAlign, noc);
            }

            {
                DeviceZoneScopedN("g6.await");
                if (role == kWaiter) {
                    noc_semaphore_wait(sem, 1);
                    noc_semaphore_set(sem, 0);
                } else {
                    noc_async_read_barrier(noc);
                    if (kSharedRead && role == kReaderPair) {
                        noc_semaphore_set(sem, 1);
                    }
                }
            }

            {
                // Permute + store, bank group by bank group: group g holds the span's pages span_page + g,
                // + g + NB, ... (consecutive slots of one bank) in page then chunk order; its write goes out as
                // soon as its image bytes are fenced, overlapping the next group's copies.
                DeviceZoneScopedN("g6.permute_store");
                g6_reshape::begin_image(noc);
                uint32_t img = image_base;
                for (uint32_t g = kGroup0; g < num_groups; g += kGroupStep) {
                    const uint32_t group_img = img;
                    uint32_t last_byte = img;
                    if constexpr (kPageAtoms) {
                        // Whole-page atoms inside one input row (e.g. [1, W] -> [W, 1]): each atom is one
                        // compile-time-sized copy; the group's pages step NB pages through the stream, so the
                        // in-row page index advances incrementally (no per-atom division).
                        const uint32_t p0 = span_page + g;
                        uint32_t q = p0 % kPagesPerRow;
                        uint32_t src_row = stage_row0 + (p0 / kPagesPerRow - r0) * kInPitch;
                        for (uint32_t j = g; j < num_pages; j += kNumBanks) {
                            const uint32_t src = src_row + q * kOutPageBytes;
                            ASSERT(src >= stage_base && src + kOutPageBytes <= stage_end);
                            g6_reshape::copy_run<kElem, kPageWords>(src, img, kOutPageBytes);
                            last_byte = img + kOutPageBytes - 1;
                            img += kAlign;
                            q += kNumBanks;
                            if (q >= kPagesPerRow) {
                                if constexpr (kNumBanks < kPagesPerRow) {
                                    q -= kPagesPerRow;
                                    src_row += kInPitch;
                                } else {
                                    src_row += (q / kPagesPerRow) * kInPitch;
                                    q %= kPagesPerRow;
                                }
                            }
                        }
                    } else {
                        for (uint32_t j = g; j < num_pages; j += kNumBanks) {
                            const uint32_t k_begin = j == 0 ? blk_chunk : 0;
                            const uint32_t k_end = j == num_pages - 1 ? last_page_atoms : kAtomsPerPage;
                            const uint32_t page_first = (span_page + j) * kWOut;
                            for (uint32_t k = k_begin; k < k_end; ++k) {
                                // Valid elements of this chunk: [pos, pos + n) of the stream, copied to the chunk's
                                // image slot one input-row piece at a time.
                                const uint32_t chunk_elem = k * kAtomElems;
                                uint32_t n = min_u32(kAtomElems, kWOut - chunk_elem);
                                const uint32_t pos = page_first + chunk_elem;
                                uint32_t r = pos / kWIn;
                                uint32_t c = pos % kWIn;
                                uint32_t dst = img;
                                while (n > 0) {
                                    const uint32_t len = min_u32(n, kWIn - c);
                                    const uint32_t src = stage_row0 + (r - r0) * kInPitch + c * kElem;
                                    ASSERT(src >= stage_base && src + len * kElem <= stage_end);
                                    g6_reshape::copy_run<kElem>(src, dst, len * kElem);
                                    dst += len * kElem;
                                    n -= len;
                                    c = 0;
                                    ++r;
                                }
                                last_byte = dst - 1;
                                img += kAlign;
                            }
                        }
                    }
                    g6_reshape::fence_image(last_byte);
                    noc_async_write(group_img, group_dst[g], img - group_img, noc);
                }
                ASSERT(img - image_base == n_atoms * kAlign);
                ASSERT(img - image_base <= kImageSlotBytes);
            }
            page = blk_page + (blk_chunk + n_atoms) / kAtomsPerPage;
            chunk = (blk_chunk + n_atoms) % kAtomsPerPage;
        } else {
            uint32_t num_copies = 0;
            {
                DeviceZoneScopedN("g6.gather");
                // Open covering read run: bytes [run_lo, run_hi) of input page run_page (both A-aligned),
                // landing at run_stage.
                uint32_t run_page = kNoRun;
                uint32_t run_lo = 0;
                uint32_t run_hi = 0;
                uint32_t run_stage = stage_base;

                auto close_run = [&]() {
                    if (run_page != kNoRun) {
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
                for (uint32_t i = 0; i < n_atoms; ++i) {
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
                    image_off += kAlign;
                    advance(bank, page, chunk);
                }
                ASSERT(image_off <= kImageSlotBytes);
                close_run();
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
