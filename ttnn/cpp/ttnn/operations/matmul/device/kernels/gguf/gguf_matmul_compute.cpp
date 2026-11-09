// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel of the GGUF matmul: a decoded weight tile goes from DEST to the matmul and never to L1
// (api/compute/experimental/gguf_decode.h, dest_srca_mm.h). Per segment of an N tile one full-sync DEST acquire holds
// the Float32 accumulator (tile 0) and two sets of (decoded tile, page, decoder scratch), tiles 1-3 and 4-6,
// alternating per K tile. The pack thread decodes K tile k + 1 while the math thread multiplies tile k
// (gguf_decode_overlap_*); the unpacker copies the next page into DEST meanwhile, before phase 2 where that phase's
// stores are not arbitrated against it (MXFP4, IQ4_NL, IQ4_XS, Q2_K). The input arrives in chunks
// (gguf_matmul_writer.cpp) and each K tile is waited for when first reached. Compile args: format id, K tiles of the
// band (c_1 holds those), input tiles per chunk, input rows the matmul computes (8, 16 or 32), the band's first K tile.
// Runtime args: segments, then per segment its first and end K tile and its role (gguf_matmul_program_factory.cpp
// Role); a finish has the count of other cores' partials in place of its first K tile. Codebook formats also look up
// part of each page's grid levels here (CodebookLookups).

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/experimental/dest_srca_mm.h"
#include "api/compute/experimental/gguf_decode.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/ckernel_gguf_page.h"
#include "experimental/gguf_page.h"
#include "internal/tt-1xx/cache.h"
#include "gguf_matmul_compute_common.hpp"

namespace {

// Codebook formats: a page's level codes are table lookups of its index words, in GGUF_LOOKUP_PARTS equal parts. The
// reader and the writer look up the first GGUF_READER_PARTS + GGUF_WRITER_PARTS before the page reaches c_0; here the
// math thread takes the next GGUF_MATH_PARTS, the unpack thread the next GGUF_UNPACK_PARTS, and the pack thread the
// rest. Math and pack run up to kLookAhead pages ahead of the decode and publish their counts of pages done in c_3; the
// unpack thread waits for both counts, as cb_wait_front waits for tiles, before it unpacks a page. Math and pack wait
// only for the reader's pages, so the waits form no cycle.
template <uint32_t fmt>
class CodebookLookups {
    static constexpr auto FMT = static_cast<experimental::gguf::Format>(fmt);
    static constexpr uint32_t kLookAhead = 2;
    static constexpr uint32_t kParts = GGUF_LOOKUP_PARTS;
    static constexpr uint32_t kMathBegin = GGUF_READER_PARTS + GGUF_WRITER_PARTS;
    static constexpr uint32_t kUnpackBegin = kMathBegin + GGUF_MATH_PARTS;
    static constexpr uint32_t kPackBegin = kUnpackBegin + GGUF_UNPACK_PARTS;
    static constexpr bool kOn = experimental::gguf::is_codebook<FMT>() && !GGUF_EXPANDED;
    static constexpr bool kMath = kOn && GGUF_MATH_PARTS > 0;
    static constexpr bool kUnpack = kOn && GGUF_UNPACK_PARTS > 0;
    static constexpr bool kPack = kOn && kPackBegin < kParts;
    static_assert(
        !kOn || ckernel::gguf::page_layout<static_cast<ckernel::gguf::Format>(fmt)>().page_tiles == 1,
        "one K tile per page");
    static_assert(!kOn || kPackBegin <= kParts, "lookup parts");
    static_assert(
        !kOn || (FMT == experimental::gguf::Format::IQ3_S || FMT == experimental::gguf::Format::IQ3_XXS ? 64 : 32) %
                        (8 * kParts) ==
                    0,
        "lookup parts of a multiple of 8 index words");
    // c_3's words: the pack thread's count at 0 and the math thread's at 4 (separate cache lines), then c_0's read
    // pointer, limit, size and page size (from the unpack thread, as the math thread has no circular buffer interfaces)
    // and c_0's acked count before the first page.
    static constexpr uint32_t kPackCount = 0, kMathCount = 4, kGeometry = 8;
    static constexpr uint32_t cb_page = tt::CBIndex::c_0, cb_sync = tt::CBIndex::c_3;

    // Index-word parts [B, E) of the page at L1 byte address `page`.
    template <uint32_t B, uint32_t E>
    static __attribute__((noinline)) void lookup_parts(const uint32_t page) {
        using namespace experimental::gguf;
        const auto* indices = reinterpret_cast<const uint32_t*>(page + index_offset<FMT>());
        for (uint32_t q = B; q < E; q++) {
            lookup_grid<FMT>(
                indices, level_table<FMT>(), reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page), q, kParts);
        }
    }

    volatile tt_l1_ptr uint32_t* sync_ = nullptr;
    uint32_t total_pages_ = 0;                            // pages of this core's K work
    uint32_t done_ = 0;                                   // math, pack: pages looked up; unpack: pages entered
    uint32_t tiles_ = 0;                                  // tiles decoded
    uint32_t ptr_ = 0, limit_ = 0, size_ = 0, step_ = 0;  // math, pack: c_0's next page to look up
    uint16_t acked0_ = 0;

    static void wait_min(volatile tt_l1_ptr uint32_t* count, const uint32_t n) {
        do {
            invalidate_l1_cache();
        } while (*count < n);
    }

    // Pages up to the count `target` looked up by this thread: parts [B, E).
    template <uint32_t B, uint32_t E>
    void run(uint32_t target, const uint32_t count_word) {
        target = target < total_pages_ ? target : total_pages_;
        while (done_ < target) {
            // The page's arrival in c_0, tested as cb_wait_front tests it.
            const uint16_t want = static_cast<uint16_t>(acked0_ + done_ + 1);
            while (static_cast<int16_t>(static_cast<uint16_t>(*get_cb_tiles_received_ptr(cb_page)) - want) < 0) {
            }
            invalidate_l1_cache();
            lookup_parts<B, E>(ptr_ << cb_addr_shift);
            asm volatile("fence" ::: "memory");  // the levels are in L1 before the count
            sync_[count_word] = ++done_;
            ptr_ += step_;
            if (ptr_ >= limit_) {
                ptr_ -= size_;
            }
        }
    }

    void receive_geometry(const uint32_t sync_addr) {
        sync_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_addr);
        invalidate_l1_cache();
        ptr_ = sync_[kGeometry];
        limit_ = sync_[kGeometry + 1];
        size_ = sync_[kGeometry + 2];
        step_ = sync_[kGeometry + 3];
        acked0_ = static_cast<uint16_t>(sync_[kGeometry + 4]);
    }

public:
    // Before the first page. Runtime args as in kernel_main.
    void start(const uint32_t segments, const uint32_t finish) {
        if constexpr (kMath || kPack) {
            for (uint32_t s = 0; s < segments; s++) {
                if (get_arg_val<uint32_t>(3 + 3 * s) != finish) {
                    total_pages_ += get_arg_val<uint32_t>(2 + 3 * s) - get_arg_val<uint32_t>(1 + 3 * s);
                }
            }
            UNPACK(({
                const uint32_t sync_addr = get_local_cb_interface(cb_sync).fifo_rd_ptr << cb_addr_shift;
                sync_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_addr);
                sync_[kPackCount] = 0;
                sync_[kMathCount] = 0;
                sync_[kGeometry] = get_local_cb_interface(cb_page).fifo_rd_ptr;
                sync_[kGeometry + 1] = get_local_cb_interface(cb_page).fifo_limit;
                sync_[kGeometry + 2] = get_local_cb_interface(cb_page).fifo_size;
                sync_[kGeometry + 3] = get_local_cb_interface(cb_page).fifo_page_size;
                sync_[kGeometry + 4] = *get_cb_tiles_acked_ptr(cb_page);
                asm volatile("fence" ::: "memory");
                if constexpr (kMath) {
                    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, sync_addr);
                }
                if constexpr (kPack) {
                    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, sync_addr);
                }
            }));
            MATH(({
                if constexpr (kMath) {
                    receive_geometry(ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId));
                }
            }));
            PACK(({
                if constexpr (kPack) {
                    receive_geometry(ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId));
                }
            }));
        }
    }

    // A segment's first page is in c_0.
    void segment_start() {
        PACK(({
            if constexpr (kPack) {
                run<kPackBegin, kParts>(tiles_ + kLookAhead, kPackCount);
            }
        }));
        MATH(({
            if constexpr (kMath) {
                run<kMathBegin, kUnpackBegin>(tiles_ + kLookAhead, kMathCount);
            }
        }));
    }

    // The pack thread decoded a tile.
    void tile_decoded() {
        if constexpr (kOn) {
            tiles_++;
        }
        PACK(({
            if constexpr (kPack) {
                run<kPackBegin, kParts>(tiles_ + kLookAhead, kPackCount);
            }
        }));
    }

    // The math thread multiplied a tile.
    void matmul_done() {
        MATH(({
            if constexpr (kMath) {
                run<kMathBegin, kUnpackBegin>(tiles_ + kLookAhead, kMathCount);
            }
        }));
    }

    // The page at c_0's front is complete before the unpack thread unpacks it.
    void before_unpack() {
        UNPACK(({
            if constexpr (kOn) {
                done_++;
            }
            if constexpr (kUnpack) {
                invalidate_l1_cache();
                lookup_parts<kUnpackBegin, kPackBegin>(get_local_cb_interface(cb_page).fifo_rd_ptr << cb_addr_shift);
                asm volatile("fence" ::: "memory");
            }
            if constexpr (kPack) {
                wait_min(sync_ + kPackCount, done_);
            }
            if constexpr (kMath) {
                wait_min(sync_ + kMathCount, done_);
            }
        }));
    }
};

}  // namespace

void kernel_main() {
    using namespace ckernel;
    constexpr uint32_t whole = 0, finish = 3;  // Role
    constexpr uint32_t fmt = get_compile_time_arg_val(0);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t chunk = get_compile_time_arg_val(2);
    constexpr uint32_t in0_rows = get_compile_time_arg_val(3);
    constexpr uint32_t k_first = get_compile_time_arg_val(4);
    constexpr gguf::PageLayout L = gguf::page_layout<static_cast<gguf::Format>(fmt)>();
    constexpr bool group_step = gguf::group_scales<fmt>;
    constexpr auto pack = GgufDecodeThread::Pack;
    constexpr uint32_t cb_page = tt::CBIndex::c_0, cb_in0 = tt::CBIndex::c_1, cb_out = gguf_matmul::cb_out;
    const uint32_t segments = get_arg_val<uint32_t>(0);
    // DEST tiles of the two sets: decoded tile, its page in the next tile, scratch past the page. MXFP4 and NVFP4:
    // decoded tiles 3 and 5, pages 4 and 6; their block-float codes go to the 16-bit rows that the decoded tile's
    // rows address (dest_srca_mm_page_unpack_codes), in tiles 1 and 2.
    constexpr bool bfp = gguf::fp4_bfp<fmt>;
    auto out_of = [](uint32_t set) -> uint32_t { return bfp ? (set ? 5 : 3) : (set ? 4 : 1); };
    auto page_of = [](uint32_t set) -> uint32_t { return bfp ? (set ? 6 : 4) : (set ? 5 : 2); };

    compute_kernel_hw_startup(cb_page, cb_in0, cb_out);
    gguf_decode_init<fmt, pack>();
    dest_srca_mm_init<in0_rows>(cb_in0, cb_page);
    gguf_decode_overlap_init();
    CodebookLookups<fmt> lookups;
    lookups.start(segments, finish);

    uint32_t in0_ready = 0;
    for (uint32_t s = 0; s < segments; s++) {
        const uint32_t k0 = get_arg_val<uint32_t>(1 + 3 * s);
        const uint32_t k1 = get_arg_val<uint32_t>(2 + 3 * s);
        const uint32_t role = get_arg_val<uint32_t>(3 + 3 * s);
        if (role == finish) {
            // k0 is the count of the other cores' partials. Finishes come last.
            dest_srca_mm_uninit();
            gguf_matmul::finish_segment(k0);
            continue;
        }

        tile_regs_acquire();
        gguf_decode_overlap_acquire();
        cb_wait_front(cb_page, 1);
        lookups.segment_start();
        lookups.before_unpack();
        if constexpr (bfp) {
            dest_srca_mm_page_to_dest_codes(
                cb_page,
                0,
                page_of(0),
                out_of(0),
                gguf::FP4_CODES,
                gguf::FP4_CODE_EXP,
                L.scale_offset,
                gguf::FP4_SCALE_WORDS);
        } else {
            dest_srca_mm_page_to_dest(cb_page, 0, page_of(0));
        }
        gguf_decode_overlap_page_ready();
        uint32_t set = 0;
        for (uint32_t kt = k0; kt < k1; kt++) {
            const uint32_t t = kt % L.group_tiles;
            const uint32_t out = out_of(set);
            const uint32_t next_page = page_of(set ^ 1);
            const bool next = kt + 1 < k1;
            gguf_decode_overlap_body_ready<fmt, false>();
            if (next) {
                if constexpr (bfp) {
                    dest_srca_mm_page_post_codes(next_page, out_of(set ^ 1));
                } else {
                    dest_srca_mm_page_post(next_page);
                }
            }
            gguf_decode_overlap_wait_page();
            if (kt == k0) {
                gguf_zero_tile<pack>(0);
            }
            // The group step for each DEST set: k0 starts a group.
            if (group_step && t < 2) {
                gguf_decode_group<fmt, pack>(out);
            }
            gguf_decode_tile<fmt, pack>(out, t);
            gguf_decode_overlap_tile_done();
            lookups.tile_decoded();
            const uint32_t in0 = kt - k_first;
            if (in0 >= in0_ready) {
                in0_ready = (in0 / chunk + 1) * chunk < k_tiles ? (in0 / chunk + 1) * chunk : k_tiles;
                cb_wait_front(cb_in0, in0_ready);
            }
            dest_srca_mm_unpack(cb_in0, in0);
            if (next) {
                if ((kt + 1) % L.page_tiles == 0) {
                    cb_pop_front(cb_page, 1);
                    cb_wait_front(cb_page, 1);
                    lookups.before_unpack();
                }
                if constexpr (bfp) {
                    dest_srca_mm_page_unpack_codes(
                        cb_page, 0, gguf::FP4_CODES, gguf::FP4_CODE_EXP, L.scale_offset, gguf::FP4_SCALE_WORDS);
                } else {
                    dest_srca_mm_page_unpack(cb_page, 0);
                }
                dest_srca_mm_page_wait(cb_page, next_page);
                gguf_decode_overlap_page_ready();
            }
            gguf_decode_overlap_body_ready<fmt, true>();
            gguf_decode_overlap_wait_tile();
            dest_srca_mm_math<in0_rows>(out, 0);
            lookups.matmul_done();
            set ^= 1;
        }
        cb_pop_front(cb_page, 1);
        gguf_matmul::pack_segment(role == whole);
    }
    dest_srca_mm_uninit();
}
