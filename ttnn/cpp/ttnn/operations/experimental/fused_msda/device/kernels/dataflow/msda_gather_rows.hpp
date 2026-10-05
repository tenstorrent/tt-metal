// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row-major gather of one sampling point's four bilinear corners, for a range of
// query rows. Shared by the reader and, when the gather is split, by the writer:
// the two data-movement RISCs each take a disjoint range of rows of the same
// staging block, so they share the issue cost while every row is still produced
// by exactly one RISC.
//
// Row r of the block is [NW | NE | SW | SE], D bf16 each (see
// fused_msda_reader_common.hpp). A row is either fully written by NoC reads
// and zero stores to disjoint slots, or zeroed whole when r >= v_rows. The
// zeroing is load-bearing: compute builds each corner's weight without bounds
// information, so a stale slot would be multiplied by a live weight. The caller
// barriers its own NoC reads; nothing here waits.

#pragma once

#include <cstdint>
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/msda_tile_layout.hpp"

namespace fused_msda_gather {

// Decodes a bf16 that holds an exact integer, with shifts only.
//
// The compute kernel floored px on the SFPU, so the value is integral by
// construction, and going through float here would put soft-float on a core
// that has no FPU. bf16 carries 8 significant bits, so every integer up to 256
// is exact; `derive_shapes` in fused_msda_device_operation.cpp rejects feature
// maps larger than that, which is what makes an in-bounds corner's index
// exact.
//
// Anything too large to decode (including inf and NaN, whose exponent field is
// 0xFF) is mapped to a magnitude no feature map can reach, so it fails the
// bounds test rather than aliasing into it through a wrapped shift.
constexpr int32_t BF16_INT_OUT_OF_RANGE = 1 << 24;

inline int32_t bf16_exact_int(uint16_t v) {
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127;
    const bool negative = (bits & 0x80000000u) != 0;
    if (exp < 0) {
        return 0;  // |v| < 1, and v is integral, so v == 0
    }
    if (exp > 23) {
        return negative ? -BF16_INT_OUT_OF_RANGE : BF16_INT_OUT_OF_RANGE;
    }
    const uint32_t mant = (bits & 0x7FFFFFu) | 0x800000u;
    const int32_t m = static_cast<int32_t>(mant >> (23 - exp));
    return negative ? -m : m;
}

// Everything one point's gather needs, as the reader hands it to the writer
// through the mailbox. Plain uint32 words so the mailbox is a flat L1 copy.
struct PointArgs {
    uint32_t x0_l1;  // compute's floored-x tile for this point
    uint32_t y0_l1;
    uint32_t block_l1;  // the input-CB block being staged
    uint32_t v_rows;
    uint32_t start_index;  // level_start of this point's level
    uint32_t width;
    uint32_t height;
    uint32_t batch;
    uint32_t head;
};
constexpr uint32_t POINT_ARGS_WORDS = sizeof(PointArgs) / sizeof(uint32_t);

// Size of the gather_mailbox CB page; kGatherMailboxNbytes in
// fused_msda_program_factory.cpp allocates it and must stay equal.
constexpr uint32_t GATHER_MAILBOX_NBYTES = 64;
static_assert(sizeof(PointArgs) <= GATHER_MAILBOX_NBYTES, "PointArgs no longer fits the gather mailbox");

// Split gather: the reader takes rows [0, SPLIT_ROW), the writer the rest. A
// tail block with v_rows <= SPLIT_ROW leaves the writer only zeroing.
//
// The reader also builds the next point's geometry tiles while the writer
// gathers, so it takes fewer rows. 15 is the measured optimum on Blackhole for
// the BEVFormer nuscenes_base shape with value in L1 (D = 32,
// test_fused_msda_perf.py); with value in DRAM, 13-16 are within run-to-run
// noise of each other. The time is not linear in the split, so re-measure
// rather than derive it. test_fused_msda_v1_masks_out_of_bounds_corners puts
// v_rows on SPLIT_ROW and SPLIT_ROW + 1 through its own copy of this value;
// change both together.
constexpr uint32_t SPLIT_ROW = 15;
static_assert(SPLIT_ROW > 0 && SPLIT_ROW < fused_msda_tile_layout::TILE_MAX_ROWS);

// Split-gather handshake. Per point, in the same (tile, level, point) order on
// both RISCs:
//   reader: post_point(pt); ready = ++seq; push the next point's geometry;
//           gather rows [0, SPLIT_ROW); barrier; wait done == seq; pop x0/y0;
//           push the block.
//   writer: wait ready == ++seq; fetch_point; gather rows [SPLIT_ROW, 32);
//           barrier; done = seq.
// Both sides must post or consume exactly NUM_LEVELS * NUM_POINTS points per
// output tile, every tile: a skipped post hangs the other side. The mailbox is
// rewritten only after done == seq, so one page is enough. These are counters
// rather than a CB because the writer fills part of a block the reader
// reserved, and x0/y0 must stay at the reader's front until the writer has
// decoded them.

inline void post_point(uint32_t mailbox_l1, const PointArgs& a) {
    const uint32_t* src = reinterpret_cast<const uint32_t*>(&a);
    CoreLocalMem<volatile uint32_t> dst(mailbox_l1);
    for (uint32_t i = 0; i < POINT_ARGS_WORDS; ++i) {
        dst[i] = src[i];
    }
}

inline PointArgs fetch_point(uint32_t mailbox_l1) {
    PointArgs a;
    uint32_t* dst = reinterpret_cast<uint32_t*>(&a);
    CoreLocalMem<volatile uint32_t> src(mailbox_l1);
    for (uint32_t i = 0; i < POINT_ARGS_WORDS; ++i) {
        dst[i] = src[i];
    }
    return a;
}

inline void zero_words(uint32_t l1_addr, uint32_t words) {
    uint32_t* p = CoreLocalMem<uint32_t>(l1_addr).get_unsafe_ptr();
#pragma GCC unroll 8
    for (uint32_t i = 0; i < words; ++i) {
        p[i] = 0;
    }
}

// Cfg carries the value layout as static constexpr members: D, NUM_KEYS,
// NUM_HEADS, VALUE_PACKED (packed (B, S, H*D) vs canonical (B, S, H, D)) and
// STICK_NBYTES. The stick must be exactly D*2 bytes (row-major staging packs
// the four slots back to back).
template <typename Cfg, typename ValueAccessor>
inline void gather_rows(
    Noc& noc, const ValueAccessor& value_acc, const PointArgs& a, uint32_t r_begin, uint32_t r_end) {
    constexpr uint32_t STICK = Cfg::STICK_NBYTES;
    constexpr uint32_t SLOT_WORDS = STICK / sizeof(uint32_t);
    constexpr uint32_t ROW_NBYTES = 4 * STICK;
    constexpr uint32_t ROW_WORDS = 4 * SLOT_WORDS;
    constexpr bool PACKED = Cfg::VALUE_PACKED;

    const int32_t w_i = static_cast<int32_t>(a.width);
    const int32_t h_i = static_cast<int32_t>(a.height);
    const uint32_t value_batch_base = a.batch * Cfg::NUM_KEYS;

    for (uint32_t r = r_begin; r < r_end; ++r) {
        const uint32_t row_l1 = a.block_l1 + r * ROW_NBYTES;
        if (r >= a.v_rows) {
            zero_words(row_l1, ROW_WORDS);
            continue;
        }
        const uint32_t col0 = fused_msda_tile_layout::tile_col0_offset(r);
        const int32_t x0 = bf16_exact_int(CoreLocalMem<volatile uint16_t>(a.x0_l1 + col0)[0]);
        const int32_t y0 = bf16_exact_int(CoreLocalMem<volatile uint16_t>(a.y0_l1 + col0)[0]);
        const bool xv[2] = {(x0 >= 0) && (x0 < w_i), (x0 + 1 >= 0) && (x0 + 1 < w_i)};
        const bool yv[2] = {(y0 >= 0) && (y0 < h_i), (y0 + 1 >= 0) && (y0 + 1 < h_i)};

        // c = 0, 1, 2, 3 is NW, NE, SW, SE, in lockstep with the four
        // corner_weight calls in msda_geometry.hpp::point.
        for (uint32_t dy = 0; dy < 2; ++dy) {
            const uint32_t pair_l1 = row_l1 + 2u * dy * STICK;
            if (!yv[dy]) {
                zero_words(pair_l1, 2 * SLOT_WORDS);
                continue;
            }
            const uint32_t cy = static_cast<uint32_t>(y0 + static_cast<int32_t>(dy));
            for (uint32_t dx = 0; dx < 2; ++dx) {
                const uint32_t slot_l1 = pair_l1 + dx * STICK;
                if (!xv[dx]) {
                    zero_words(slot_l1, SLOT_WORDS);
                    continue;
                }
                const uint32_t cx = static_cast<uint32_t>(x0 + static_cast<int32_t>(dx));
                const uint32_t s = a.start_index + cy * a.width + cx;
                uint32_t page;
                uint32_t offset_bytes;
                if constexpr (PACKED) {
                    page = value_batch_base + s;
                    offset_bytes = a.head * (Cfg::D * 2u);
                } else {
                    page = (value_batch_base + s) * Cfg::NUM_HEADS + a.head;
                    offset_bytes = 0;
                }
                noc.async_read(
                    value_acc,
                    CoreLocalMem<uint32_t>(slot_l1),
                    STICK,
                    {.page_id = page, .offset_bytes = offset_bytes},
                    {.offset_bytes = 0});
            }
        }
    }
}

}  // namespace fused_msda_gather
