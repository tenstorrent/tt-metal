// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Block walk of the Quasar permute kernel (see permute_plan_layout.hpp for the plan).
//
// Generic over the plan container and over an IO policy: the DM kernel supplies the NoC transfers and its L1
// windows, a host-side check can supply plain memory. IO provides:
//   void read(page, offset_bytes, len_bytes, in_window_offset_bytes);   // input page run -> read window
//   void read_barrier();
//   void write(page, offset_bytes, len_bytes, from_read_window, window_offset_bytes);  // window -> output page run
//   void write_barrier();
//   void publish();                  // make the thread's element stores visible to the writes that follow
//   uint8_t* in_window();            // element-addressable views of the two windows
//   uint8_t* out_window();

#include <cstdint>

#include "ttnn/operations/experimental/quasar/permute/device/kernels/permute_plan_layout.hpp"

namespace ttnn::prim::qsr::permute_engine {

using namespace ttnn::prim::qsr::permute_plan;

constexpr uint32_t kIn = 0;
constexpr uint32_t kOut = 1;

// In-page geometry of an axis role: element offset of in-page index u is (u / 16) * hi + (u % 16) * lo, which
// covers both a plain run (hi = 16 * lo) and the two 16-element faces a tile has along each dimension.
struct KindGeometry {
    uint32_t page_shift;
    uint32_t lo;
    uint32_t hi;
    uint32_t mask;
};

constexpr KindGeometry kind_geometry(uint32_t kind) {
    switch (kind) {
        case kRowElem: return {0, 1, 16, ~0u};
        case kTileRow: return {5, 16, 512, kTileDim - 1};
        case kTileCol: return {5, 1, 256, kTileDim - 1};
        default: return {0, 0, 0, 0};
    }
}

constexpr uint32_t two_level(uint32_t u, uint32_t lo, uint32_t hi) { return (u >> 4) * hi + (u & 15) * lo; }

template <uint32_t E>
struct ElemType;
template <>
struct ElemType<1> {
    using type = uint8_t;
};
template <>
struct ElemType<2> {
    using type = uint16_t;
};
template <>
struct ElemType<4> {
    using type = uint32_t;
};

// The DM hart's stack is small (about 1.7 KB with the firmware's own frames on it), so the per-block state is
// kept compact and the phases of a block are separate, non-inlined frames.
#define PERMUTE_ENGINE_NOINLINE __attribute__((noinline))

// One axis of the current block, per side: how many indices the block covers, and how index t of the chunk maps
// to pages / window slots / window element offsets.
struct BlockAxis {
    uint32_t n;
    uint32_t page_stride[2];
    uint32_t lo[2];
    uint16_t slot_stride[2];  // non-zero: the axis enumerates pages on that side
    uint16_t hi_tile[2];      // non-zero: the face stride of a tile axis (else hi = 16 * lo)
    uint8_t u0[2];            // in-page index of the chunk start, for a tile axis whose page sits whole in the window

    uint32_t hi(uint32_t side) const { return hi_tile[side] != 0 ? hi_tile[side] : 16 * lo[side]; }
};

// What the transfers of the current block need besides the axes.
struct BlockTransfers {
    uint32_t base_page[2];
    uint32_t offset_bytes[2];
    uint32_t len_bytes[2];
    uint32_t footprint_elems[2];  // in-page elements the block covers per page
};

// Calls f(page_delta, slot) for every page the block touches on `side`.
template <typename F>
inline void for_each_slot(const BlockAxis* axes, uint32_t rank, uint32_t side, F&& f) {
    uint32_t count[kMaxRank] = {};
    while (true) {
        uint32_t page = 0;
        uint32_t slot = 0;
        for (uint32_t d = 0; d < rank; ++d) {
            page += count[d] * axes[d].page_stride[side];
            slot += count[d] * axes[d].slot_stride[side];
        }
        f(page, slot);
        int32_t d = static_cast<int32_t>(rank) - 1;
        for (; d >= 0; --d) {
            if (axes[d].slot_stride[side] == 0) {
                continue;
            }
            if (++count[d] < axes[d].n) {
                break;
            }
            count[d] = 0;
        }
        if (d < 0) {
            return;
        }
    }
}

// Lays out block `block`: per-axis chunk and strides, and the page / in-page run of its transfers.
template <uint32_t E, typename Plan>
PERMUTE_ENGINE_NOINLINE void decode_block(const Plan& plan, uint32_t block, BlockAxis* axes, BlockTransfers& xfer) {
    const uint32_t rank = plan[kRank];
    const uint32_t flags = plan[kFlags];
    const bool seg[2] = {(flags & kFlagInSeg) != 0, (flags & kFlagOutSeg) != 0};
    const uint32_t slot_elems[2] = {plan[kInSlotElems], plan[kOutSlotElems]};
    uint32_t seg_off[2] = {0, 0};
    xfer.base_page[kIn] = xfer.base_page[kOut] = 0;
    xfer.footprint_elems[kIn] = xfer.footprint_elems[kOut] = 1;

    uint32_t rem = block;
    for (int32_t d = static_cast<int32_t>(rank) - 1; d >= 0; --d) {
        const uint32_t a = kHeaderWords + static_cast<uint32_t>(d) * kAxisWords;
        const uint32_t extent = plan[a + kAxisExtent];
        const uint32_t chunk = plan[a + kAxisChunk];
        const uint32_t num_chunks = (extent + chunk - 1) / chunk;
        const uint32_t c0 = (rem % num_chunks) * chunk;
        rem /= num_chunks;
        BlockAxis& ax = axes[d];
        ax.n = extent - c0 < chunk ? extent - c0 : chunk;
        for (uint32_t s = 0; s < 2; ++s) {
            const uint32_t kind = plan[a + (s == kIn ? kAxisInKind : kAxisOutKind)];
            const uint32_t page_stride = plan[a + (s == kIn ? kAxisInPageStride : kAxisOutPageStride)];
            const uint32_t slot_stride = plan[a + (s == kIn ? kAxisInSlotStride : kAxisOutSlotStride)];
            const KindGeometry g = kind_geometry(kind);
            xfer.base_page[s] += (c0 >> g.page_shift) * page_stride;
            ax.page_stride[s] = page_stride;
            ax.slot_stride[s] = static_cast<uint16_t>(slot_stride);
            ax.hi_tile[s] = 0;
            ax.u0[s] = 0;
            if (kind == kOuter) {
                ax.lo[s] = slot_stride * slot_elems[s];
            } else {
                const uint32_t u = c0 & g.mask;
                seg_off[s] += two_level(u, g.lo, g.hi);
                xfer.footprint_elems[s] *= ax.n;
                ax.lo[s] = g.lo;
                if (!seg[s]) {
                    // Whole page in the window: the element sits at its in-page offset.
                    ax.hi_tile[s] = static_cast<uint16_t>(g.hi == 16 * g.lo ? 0 : g.hi);
                    ax.u0[s] = static_cast<uint8_t>(u);
                }
            }
        }
    }
    for (uint32_t s = 0; s < 2; ++s) {
        xfer.offset_bytes[s] = seg[s] ? seg_off[s] * E : 0;
        xfer.len_bytes[s] = (seg[s] ? xfer.footprint_elems[s] : plan[s == kIn ? kInPageElems : kOutPageElems]) * E;
    }
}

template <typename IO>
PERMUTE_ENGINE_NOINLINE void read_block(
    IO& io, const BlockAxis* axes, uint32_t rank, const BlockTransfers& xfer, uint32_t slot_bytes) {
    for_each_slot(axes, rank, kIn, [&](uint32_t page, uint32_t slot) {
        io.read(xfer.base_page[kIn] + page, xfer.offset_bytes[kIn], xfer.len_bytes[kIn], slot * slot_bytes);
    });
    io.read_barrier();
}

template <typename IO>
PERMUTE_ENGINE_NOINLINE void write_block(
    IO& io, const BlockAxis* axes, uint32_t rank, const BlockTransfers& xfer, uint32_t slot_bytes, bool direct) {
    for_each_slot(axes, rank, kOut, [&](uint32_t page, uint32_t slot) {
        io.write(xfer.base_page[kOut] + page, xfer.offset_bytes[kOut], xfer.len_bytes[kOut], direct, slot * slot_bytes);
    });
    io.write_barrier();
}

template <uint32_t E>
PERMUTE_ENGINE_NOINLINE void fill_window(uint8_t* window, uint32_t elems, uint32_t pad_word) {
    using Elem = typename ElemType<E>::type;
    volatile Elem* out = reinterpret_cast<volatile Elem*>(window);
    const Elem pad = static_cast<Elem>(pad_word);
    for (uint32_t i = 0; i < elems; ++i) {
        out[i] = pad;
    }
}

// Copies the block's elements from the read window to the write window. `inner` is walked by the innermost loop
// in runs that stay inside one 16-element face on both sides, so the run body is a fixed-stride copy.
template <uint32_t E>
PERMUTE_ENGINE_NOINLINE void reorder_block(
    const BlockAxis* axes, uint32_t rank, uint32_t inner, uint8_t* in_base, uint8_t* out_base) {
    using Elem = typename ElemType<E>::type;
    const volatile Elem* in = reinterpret_cast<const volatile Elem*>(in_base);
    volatile Elem* out = reinterpret_cast<volatile Elem*>(out_base);

    const BlockAxis& ia = axes[inner];
    const uint32_t lo_in = ia.lo[kIn];
    const uint32_t lo_out = ia.lo[kOut];
    const bool in_plain = ia.hi_tile[kIn] == 0;
    const bool out_plain = ia.hi_tile[kOut] == 0;
    const uint32_t face_step_in = ia.hi(kIn) - 16 * lo_in;
    const uint32_t face_step_out = ia.hi(kOut) - 16 * lo_out;

    uint32_t t[kMaxRank] = {};
    while (true) {
        uint32_t pi = 0;
        uint32_t po = 0;
        for (uint32_t d = 0; d < rank; ++d) {
            pi += two_level(axes[d].u0[kIn] + t[d], axes[d].lo[kIn], axes[d].hi(kIn));
            po += two_level(axes[d].u0[kOut] + t[d], axes[d].lo[kOut], axes[d].hi(kOut));
        }

        uint32_t ui = ia.u0[kIn];
        uint32_t uo = ia.u0[kOut];
        uint32_t remaining = ia.n;
        while (remaining > 0) {
            uint32_t run = remaining;
            if (!in_plain && 16 - (ui & 15) < run) {
                run = 16 - (ui & 15);
            }
            if (!out_plain && 16 - (uo & 15) < run) {
                run = 16 - (uo & 15);
            }
            for (uint32_t r = 0; r < run; ++r) {
                out[po] = in[pi];
                pi += lo_in;
                po += lo_out;
            }
            ui += run;
            uo += run;
            remaining -= run;
            if (!in_plain && (ui & 15) == 0) {
                pi += face_step_in;
            }
            if (!out_plain && (uo & 15) == 0) {
                po += face_step_out;
            }
        }

        int32_t d = static_cast<int32_t>(rank) - 1;
        for (; d >= 0; --d) {
            if (static_cast<uint32_t>(d) == inner) {
                continue;
            }
            if (++t[d] < axes[d].n) {
                break;
            }
            t[d] = 0;
        }
        if (d < 0) {
            return;
        }
    }
}

// Runs blocks first, first + step, ... < end.
template <uint32_t E, typename Plan, typename IO>
inline void run_blocks(const Plan& plan, IO& io, uint32_t first, uint32_t end, uint32_t step) {
    const uint32_t rank = plan[kRank];
    const uint32_t flags = plan[kFlags];
    const bool direct = (flags & kFlagDirect) != 0;
    const uint32_t in_slot_bytes = plan[kInSlotElems] * E;
    const uint32_t out_slot_bytes = plan[kOutSlotElems] * E;

    BlockAxis axes[kMaxRank];
    BlockTransfers xfer;

    for (uint32_t block = first; block < end; block += step) {
        decode_block<E>(plan, block, axes, xfer);
        read_block(io, axes, rank, xfer, in_slot_bytes);
        if (!direct) {
            if ((flags & kFlagFill) != 0 && xfer.footprint_elems[kOut] < plan[kOutPageElems]) {
                fill_window<E>(io.out_window(), plan[kOutBufBytes] / E, plan[kPadWord]);
            }
            reorder_block<E>(axes, rank, plan[kInnerAxis], io.in_window(), io.out_window());
            io.publish();
        }
        write_block(io, axes, rank, xfer, out_slot_bytes, direct);
    }
}

}  // namespace ttnn::prim::qsr::permute_engine
