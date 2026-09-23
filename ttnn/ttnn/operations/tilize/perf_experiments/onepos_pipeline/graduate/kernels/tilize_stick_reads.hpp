// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared dataflow pieces of tilize's `load_block` / `store_block` (op_design.md):
// the per-core walk and the pipelined stick-segment producer. One source of truth
// for the traversal, so the reader, the writer and (split reader) the writer's odd
// loads cannot drift apart.
//
// Walk: a Tensix core visits its rectangle column block by column block; inside a
// column block it visits its core_row_tiles tile-rows starting at `row_rotation`
// (mod core_row_tiles). The linear position in that walk is the sequence index;
// compute consumes positions in order. Under the split reader, even positions are
// produced by NCRISC into cb_input_sticks and odd positions by BRISC into
// cb_input_sticks_odd (each CB has exactly one producer).
//
// DRAM bank spreading: with a contiguous row split every Tensix core sits at the
// same local step at the same time, so un-rotated walks send all cores to
// indices congruent modulo a small factor of the bank count (3 of 12 banks on
// Wormhole). Rotating each core's walk start (and the stick order inside a
// tile-row) by a per-core amount spreads concurrent requests over the banks.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace tilize_dataflow {

// ---------------------------------------------------------------------------
// Padding (op_design.md -> load_block, pad refinement)
// ---------------------------------------------------------------------------
//
// The output tile grid is the PADDED shape's: R = prod(P[:-2]) * P[-2] / tile_h tile-rows,
// C = P[-1] / 32 tile-columns. Output tile-row r belongs to padded image r / rows_per_image;
// its sticks are rows h0 .. h0 + tile_h - 1 of that image (h0 = (r % rows_per_image) * tile_h).
// A stick exists in the input iff its image exists there (every leading index < the input's
// dim) and h < H; of an existing stick, the first in_row_bytes bytes are data. Everything else
// of the tile-row (W tail, H tail, whole pad sticks / tile-rows / images) holds the fill value.

// Where one output tile-row's data comes from.
struct RowSource {
    uint32_t first_stick;   // input stick index of the tile-row's first stick (valid if valid_sticks > 0)
    uint32_t valid_sticks;  // leading sticks of the tile-row that exist in the input (0..tile_h)
    uint32_t data_bytes;    // bytes of the segment each existing stick carries (0..segment bytes)
};

// The per-image stick map r -> (image, h). Leading dims: the host passes only the innermost
// `num_lead` padded dims up to the outermost one the pad grows (innermost first); every outer
// leading dim is equal on both sides, so it folds into the remaining quotient unchanged.
template <uint32_t tile_h, uint32_t tile_col_bytes>
struct PadMap {
    static constexpr uint32_t MAX_LEAD_DIMS = 8;
    uint32_t rows_per_image;  // padded tile-rows per padded image: P[-2] / tile_h
    uint32_t in_h;            // input rows per image: X[-2]
    uint32_t in_row_bytes;    // data bytes per input stick: X[-1] * elem_bytes
    uint32_t num_lead;
    uint32_t lead_padded[MAX_LEAD_DIMS];
    uint32_t lead_input[MAX_LEAD_DIMS];

    // RT args at `base`: rows_per_image, in_h, in_row_bytes, num_lead, then num_lead
    // (padded, input) dim pairs, innermost first.
    explicit PadMap(uint32_t base) :
        rows_per_image(get_arg_val<uint32_t>(base)),
        in_h(get_arg_val<uint32_t>(base + 1)),
        in_row_bytes(get_arg_val<uint32_t>(base + 2)),
        num_lead(get_arg_val<uint32_t>(base + 3)) {
        for (uint32_t d = 0; d < num_lead; ++d) {
            lead_padded[d] = get_arg_val<uint32_t>(base + 4 + 2 * d);
            lead_input[d] = get_arg_val<uint32_t>(base + 5 + 2 * d);
        }
    }

    FORCE_INLINE RowSource source(uint32_t row, uint32_t first_col, uint32_t valid_width) const {
        const uint32_t image = row / rows_per_image;
        const uint32_t h0 = (row - image * rows_per_image) * tile_h;
        uint32_t rem = image;
        uint32_t in_image = 0;
        uint32_t in_stride = 1;
        bool exists = true;
        for (uint32_t d = 0; d < num_lead; ++d) {
            const uint32_t q = rem / lead_padded[d];
            const uint32_t idx = rem - q * lead_padded[d];
            exists = exists && idx < lead_input[d];
            in_image += idx * in_stride;
            in_stride *= lead_input[d];
            rem = q;
        }
        in_image += rem * in_stride;
        RowSource src{0, 0, 0};
        if (exists && h0 < in_h) {
            const uint32_t left = in_h - h0;
            src.valid_sticks = left < tile_h ? left : tile_h;
            src.first_stick = in_image * in_h + h0;
            const uint32_t offset = first_col * tile_col_bytes;
            const uint32_t segment = valid_width * tile_col_bytes;
            if (offset < in_row_bytes) {
                const uint32_t avail = in_row_bytes - offset;
                src.data_bytes = avail < segment ? avail : segment;
            }
        }
        return src;
    }
};

// Fills L1 byte ranges with the pad value. Short ranges are CPU-stored
// (dataflow_kernel_lib::fill_l1_range); long ones are copied from a reader-private
// source region pre-filled once (`src_bytes`, in its own CB) by NoC loopback reads on this
// Tensix core, so the RISC-V only issues commands. The loopback reads carry their own
// transaction id (`trid`); complete() waits for them. Every range is a multiple of
// elem_bytes and starts on an element boundary.
template <uint32_t elem_bytes, uint32_t src_bytes, uint32_t noc_min_bytes, uint32_t trid>
struct PadFill {
    static_assert(src_bytes % L1_ALIGNMENT == 0 && src_bytes > 0, "pad source: whole NoC-aligned units");
    static_assert(noc_min_bytes >= 2 * L1_ALIGNMENT, "a NoC fill needs an aligned interior");
    static constexpr uint32_t MAX_READS_PER_TRID = 255;  // NOC_MAX_TRANSACTION_ID_COUNT
    uint32_t value;
    uint64_t src_noc_addr;
    uint32_t reads_in_flight = 0;

    PadFill(uint32_t cb_pad_source, uint32_t value_) : value(value_) {
        const uint32_t src = get_write_ptr(cb_pad_source);
        dataflow_kernel_lib::fill_l1_range<elem_bytes>(src, src_bytes, value);
        src_noc_addr = get_noc_addr(src);
        asm volatile("" ::: "memory");  // the source stores precede every loopback read of it
    }

    // Fill [dst, dst + bytes). May leave loopback reads in flight under `trid` and leaves `trid`
    // as the NoC read state: the caller re-selects its own read trid afterwards and calls
    // complete() before publishing the destination.
    FORCE_INLINE void fill(uint32_t dst, uint32_t bytes) {
        if (bytes < noc_min_bytes) {
            dataflow_kernel_lib::fill_l1_range<elem_bytes>(dst, bytes, value);
            return;
        }
        const uint32_t head = (L1_ALIGNMENT - (dst % L1_ALIGNMENT)) % L1_ALIGNMENT;
        dataflow_kernel_lib::fill_l1_range<elem_bytes>(dst, head, value);
        dst += head;
        bytes -= head;
        const uint32_t tail = bytes % L1_ALIGNMENT;
        uint32_t bulk = bytes - tail;
        noc_async_read_set_trid(trid);
        while (bulk > 0) {
            if (reads_in_flight == MAX_READS_PER_TRID) {
                complete();
            }
            const uint32_t chunk = bulk < src_bytes ? bulk : src_bytes;
            noc_async_read(src_noc_addr, dst, chunk);
            ++reads_in_flight;
            dst += chunk;
            bulk -= chunk;
        }
        dataflow_kernel_lib::fill_l1_range<elem_bytes>(dst, tail, value);
    }

    // Fill `count` ranges [dst + i * stride, dst + i * stride + bytes): the W tail of a
    // tile-row's existing sticks. `stride` (the stick stride) is a multiple of 4, so every range
    // shares dst's alignment: a short band is split into head elements / words / tail elements
    // ONCE and stored with tight loops (fill_l1_range re-derives that split per call, which cost
    // more than the stores on a narrow W tail); a long band goes range by range to fill().
    FORCE_INLINE void fill_band(uint32_t dst, uint32_t stride, uint32_t count, uint32_t bytes) {
        if (bytes >= noc_min_bytes) {
            for (uint32_t i = 0; i < count; ++i, dst += stride) {
                fill(dst, bytes);
            }
            return;
        }
        uint32_t head = (4 - (dst & 3)) & 3;
        head = head < bytes ? head : bytes;
        const uint32_t words = (bytes - head) / 4;
        const uint32_t tail = bytes - head - 4 * words;
        const uint32_t word = elem_bytes == 4   ? value
                              : elem_bytes == 2 ? ((value & 0xFFFF) | (value << 16))
                                                : (value & 0xFF) * 0x01010101u;
        for (uint32_t i = 0; i < count; ++i, dst += stride) {
            if (head != 0) {
                dataflow_kernel_lib::fill_l1_range<elem_bytes>(dst, head, value);
            }
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + head);
            for (uint32_t k = 0; k < words; ++k) {
                w[k] = word;
            }
            if (tail != 0) {
                dataflow_kernel_lib::fill_l1_range<elem_bytes>(dst + head + 4 * words, tail, value);
            }
        }
    }

    // Every loopback fill issued so far has landed.
    FORCE_INLINE void complete() {
        if (reads_in_flight != 0) {
            noc_async_read_barrier_with_trid(trid);
            reads_in_flight = 0;
        }
    }
};

// No padding: the producer never calls into it.
struct NoPadFill {
    FORCE_INLINE void complete() {}
};

// Bytes [offset, offset + bytes) of logical stick `stick` on a paged input (a WIDTH / BLOCK /
// ND-sharded Layout::ROW_MAJOR tensor cuts every stick into pages of the shard width), one NoC
// read per page the range overlaps.
template <uint32_t page_bytes, uint32_t pages_per_stick, typename Accessor>
FORCE_INLINE void read_paged_segment(
    const Accessor& accessor, uint32_t stick, uint32_t offset, uint32_t l1_dst, uint32_t bytes) {
    uint32_t page = stick * pages_per_stick + offset / page_bytes;
    uint32_t in_page = offset % page_bytes;
    while (bytes > 0) {
        const uint32_t chunk = bytes < page_bytes - in_page ? bytes : page_bytes - in_page;
        noc_async_read(accessor.get_noc_addr(page, in_page), l1_dst, chunk);
        l1_dst += chunk;
        bytes -= chunk;
        ++page;
        in_page = 0;
    }
}

// The stick-segment reads of ONE tile-row, sequence steps s in [s_begin, s_end) of its tile_h
// sticks: step s reads stick (s + stick_rotation) mod tile_h (the per-core rotated order) into
// its tilize position l1_base + stick * block_stick_bytes. The single source of a tile-row's
// stick reads for both of its producers: StickProducer (NCRISC, steps [0, tile_h - co_read)) and,
// in the co-read regime, the writer (BRISC, steps [tile_h - co_read, tile_h)).
template <
    uint32_t tile_h,
    uint32_t block_stick_bytes,
    uint32_t page_bytes,
    uint32_t pages_per_stick,
    uint32_t noc_split = 0,
    typename Accessor>
FORCE_INLINE void read_tile_row_sticks(
    const Accessor& accessor,
    uint32_t first_stick,
    uint32_t l1_base,
    uint32_t segment_offset,
    uint32_t segment_bytes,
    uint32_t stick_rotation,
    uint32_t s_begin,
    uint32_t s_end) {
    for (uint32_t s = s_begin; s < s_end; ++s) {
        const uint32_t stick = (s + stick_rotation) & (tile_h - 1);
        const uint32_t l1_dst = l1_base + stick * block_stick_bytes;
        if constexpr (pages_per_stick == 1) {
            const uint8_t noc = (noc_split != 0 && (s % noc_split) == noc_split - 1) ? 1 - noc_index : noc_index;
            noc_async_read(accessor.get_noc_addr(first_stick + stick, segment_offset, noc), l1_dst, segment_bytes, noc);
        } else {
            read_paged_segment<page_bytes, pages_per_stick>(
                accessor, first_stick + stick, segment_offset, l1_dst, segment_bytes);
        }
    }
}

// Co-read stick lists (Perf 2, hop_aware_coread). Which sticks of the tile-row each RISC-V reads is
// chosen on the host by DRAM-bank geometry (tilize_program_descriptor._co_read_lists): a read's DATA
// path is bank -> core, east/south on NoC0 but west/north on NoC1, so the same bank can be a few
// hops or most of the torus away depending on the NoC. The positional R8 cut sent half the sticks
// down the long NoC1 path (BRISC's 16 reads took 2x NCRISC's 32 at 1 KiB segments).
//
// The host cannot see physical Tensix coordinates (the bindings return translated ones, and WH
// row harvesting shifts the physical row), so it precomputes the lists for each physical row this
// Tensix core can be on and the kernel picks by its own NoC0 node id. RT layout at `arg`: one word
// of candidate physical NoC0 rows (row c in byte c, up to 3; unused bytes 0xFF), then one list block
// per candidate of 1 + ceil(tile_h / 4) words: n, then n stick indices (0..tile_h-1) packed 4 per
// word, low byte first, in issue order. Both RISC-Vs read the same NoC0 register, so they always
// pick complementary lists. Compiled in only under CO_READ_LISTED (the host keeps R8's positional
// cut, with HEAD's exact binaries, where its model predicts less than the list path's own overhead
// to gain: CO_READ_LIST_MIN_GAIN_CYCLES). Raw register read (NOC_CMD_BUF_READ_REG(NOC_NODE_ID)): no dataflow API
// returns the physical coordinate.
template <uint32_t tile_h>
FORCE_INLINE uint32_t select_co_read_list(uint32_t arg) {
    constexpr uint32_t block_words = 1 + (tile_h + 3) / 4;
    const uint32_t rows = get_arg_val<uint32_t>(arg);
    const uint32_t my_row = (NOC_CMD_BUF_READ_REG(0, 0, NOC_NODE_ID) >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    const uint32_t j = ((rows >> 8) & 0xFF) == my_row ? 1 : (((rows >> 16) & 0xFF) == my_row ? 2 : 0);
    return arg + 1 + j * block_words;
}

// The stick reads of ONE tile-row named by a co-read list (select_co_read_list). The packed words
// are loaded once per 4 sticks and unpacked in registers, so the per-read work matches
// read_tile_row_sticks' (measured: a per-read L1 byte load costs ~8 cycles per read, a bit-mask walk
// ~5.5 cycles per step -- both visible on the issue-bound one-position walks).
template <uint32_t tile_h, uint32_t block_stick_bytes, uint32_t page_bytes, uint32_t pages_per_stick, typename Accessor>
FORCE_INLINE void read_tile_row_sticks_listed(
    const Accessor& accessor,
    uint32_t first_stick,
    uint32_t l1_base,
    uint32_t segment_offset,
    uint32_t segment_bytes,
    uint32_t list_arg) {
    auto read_one = [&](uint32_t stick) {
        const uint32_t l1_dst = l1_base + stick * block_stick_bytes;
        if constexpr (pages_per_stick == 1) {
            noc_async_read(accessor.get_noc_addr(first_stick + stick, segment_offset), l1_dst, segment_bytes);
        } else {
            read_paged_segment<page_bytes, pages_per_stick>(
                accessor, first_stick + stick, segment_offset, l1_dst, segment_bytes);
        }
    };
    const uint32_t n = get_arg_val<uint32_t>(list_arg);
    const uint32_t full = n >> 2;
    for (uint32_t w = 0; w < full; ++w) {
        uint32_t word = get_arg_val<uint32_t>(list_arg + 1 + w);
#pragma GCC unroll 4
        for (uint32_t k = 0; k < 4; ++k, word >>= 8) {
            read_one(word & 0xFF);
        }
    }
    if ((n & 3) != 0) {
        uint32_t word = get_arg_val<uint32_t>(list_arg + 1 + full);
        for (uint32_t k = 0; k < (n & 3); ++k, word >>= 8) {
            read_one(word & 0xFF);
        }
    }
}

// Co-read (Refinement 8, CO_READ_SHARE): on a Tensix core whose whole walk is ONE position, the
// writer RISC-V (BRISC, NoC1) would sit idle until compute finishes, so it issues the last
// co_read sticks of the tile-row straight into cb_input_sticks' first slot while NCRISC issues
// the rest; one read-issuing RISC-V is the latency chain there (~40 cycles per stick read).
// NCRISC stays the CB's only producer (reserve / push); BRISC only fills bytes of the slot, then
// raises a local L1 flag (a program semaphore) after its read barrier. This is StickProducer's
// Fill for that walk: complete() runs before the slot is published, waits for the flag and
// re-arms it for the next launch.
struct CoReadLanded {
    volatile tt_l1_ptr uint32_t* flag;
    explicit CoReadLanded(uint32_t flag_addr) : flag(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr)) {}
    FORCE_INLINE void complete() {
        MaybeDeviceZoneScope("reader_coread_wait");  // NCRISC waiting on BRISC's share landing
        noc_semaphore_wait(flag, 1);
        noc_semaphore_set(flag, 0);
    }
};

// One position of the per-core walk.
template <uint32_t block_width>
struct Walker {
    uint32_t row_start, core_row_tiles, col_start, core_col_tiles, row_rotation;
    uint32_t col_block_idx = 0;
    uint32_t i = 0;          // index within the column block's tile-row walk
    uint32_t row_local = 0;  // rotated local tile-row

    Walker(
        uint32_t row_start_,
        uint32_t core_row_tiles_,
        uint32_t col_start_,
        uint32_t core_col_tiles_,
        uint32_t rotation) :
        row_start(row_start_),
        core_row_tiles(core_row_tiles_),
        col_start(col_start_),
        core_col_tiles(core_col_tiles_),
        row_rotation(rotation % core_row_tiles_),
        row_local(rotation % core_row_tiles_) {}

    uint32_t num_positions() const { return core_row_tiles * ((core_col_tiles + block_width - 1) / block_width); }
    uint32_t row() const { return row_start + row_local; }
    uint32_t first_col() const { return col_start + col_block_idx * block_width; }
    uint32_t valid_width() const {
        const uint32_t remaining = core_col_tiles - col_block_idx * block_width;
        return remaining < block_width ? remaining : block_width;
    }
    void advance() {
        if (++row_local == core_row_tiles) {
            row_local = 0;
        }
        if (++i == core_row_tiles) {
            i = 0;
            row_local = row_rotation;
            ++col_block_idx;
        }
    }
};

// Pipelined producer of CB slots in one input CB. A slot holds `rows_per_slot`
// consecutive walk positions (tile-rows of block_width pages each), so the read
// barrier and the CB push happen once per slot, not once per tile-row
// (`rows_per_slot` = the host-derived rows_per_quantum, see QUANTUM_MIN_TILES).
//
// issue_row(): opening a slot reserves room for (outstanding + 1) slots and tags
//              the slot's reads with its NoC transaction id; if `read_ahead` slots
//              are already outstanding, the oldest is completed first. Then the
//              tile_h stick-segment reads of one tile-row land in the open slot.
// complete_oldest(): barrier on the oldest slot's transaction id, push its pages.
//
// Every push is the nominal rows_per_slot * block_width pages except the very
// last one of the kernel (a partial final slot), so the CB write pointer after
// n full pushes is base + (n % depth) * slot_bytes; slot addresses are derived
// from that instead of CB internals. Nothing is pushed after the partial slot,
// so it cannot break the ring-wrap invariant. Requires read_ahead <= depth and
// depth <= 15 (one trid per slot). The host keeps a slot's reads within one
// trid's outstanding-count range (NOC_MAX_TRANSACTION_ID_COUNT).
//
// Input pages: interleaved and HEIGHT-sharded Layout::ROW_MAJOR tensors keep one
// page per stick (pages_per_stick == 1: one NoC read per stick segment). A
// WIDTH / BLOCK / ND-sharded one cuts each stick into pages of the shard width
// (`page_bytes` data bytes each; page id = stick * pages_per_stick + k), so a
// segment is read as one NoC read per page it overlaps (read_paged_segment).
//
// Parked perf knobs (host tilize_program_descriptor.py documents the measurements;
// each default compiles to the plain path above):
//   noc_split     every noc_split-th stick read of a tile-row goes out on the other NoC
//                 (0: all on noc_index). Needs the kernel in DM_DYNAMIC_NOC mode.
//   eager_publish issue_row() first pushes every sealed slot whose reads have already
//                 landed (non-blocking trid poll), so read_ahead never delays a push.
//   stride_banks  > 0 on a DRAM-interleaved input: stick p + stride_banks lives in the
//                 same bank as stick p, one aligned page further, so per core only
//                 stride_banks addresses come from the accessor (prime()) and every other
//                 stick address is one of them plus a multiple of the page stride.
//   bank_major    (with stride_banks) issue a tile-row's sticks bank by bank, so each
//                 bank sees consecutive addresses back to back.
template <
    uint32_t cb,
    uint32_t block_width,
    uint32_t depth,
    uint32_t read_ahead,
    uint32_t tile_h,
    uint32_t tile_col_bytes,
    uint32_t rows_per_slot,
    uint32_t page_bytes,
    uint32_t pages_per_stick,
    uint32_t noc_split = 0,
    bool eager_publish = false,
    uint32_t stride_banks = 0,
    bool bank_major = false,
    uint32_t co_read = 0,
    bool co_read_listed = false>
struct StickProducer {
    static_assert((tile_h & (tile_h - 1)) == 0, "tile_h must be a power of two");
    static_assert(read_ahead >= 1 && read_ahead <= depth, "read_ahead must be in [1, depth]");
    static_assert(depth <= 15, "one NoC transaction id per CB slot");
    static_assert(rows_per_slot >= 1, "a slot holds at least one tile-row");
    static_assert(stride_banks == 0 || pages_per_stick == 1, "bank-stride addressing needs one page per stick");
    static_assert(!bank_major || stride_banks != 0, "bank-major order needs bank-stride addressing");
    static_assert(co_read < tile_h && (co_read == 0 || stride_banks == 0), "co-read: the plain stick order");
    static constexpr uint32_t in_tile_bytes = tile_h * tile_col_bytes;  // tile-sized page: tile_h stick segments
    static constexpr uint32_t row_pages = block_width;                  // pages per tile-row
    static constexpr uint32_t row_bytes = block_width * in_tile_bytes;
    static constexpr uint32_t slot_pages = rows_per_slot * row_pages;
    static constexpr uint32_t slot_bytes = rows_per_slot * row_bytes;
    static constexpr uint32_t block_stick_bytes = block_width * tile_col_bytes;  // nominal L1 stride per stick

    uint32_t base_addr;
    uint32_t stick_rotation;
    uint32_t next_slot = 0;    // slot index of the next slot to open (monotonic)
    uint32_t outstanding = 0;  // opened, not yet pushed
    uint32_t open_rows = 0;    // tile-rows already issued into the newest slot (0 = no slot open)
    uint32_t open_base = 0;    // L1 address of the newest slot
    // stride_banks: NoC addresses of this core's first stride_banks sticks, the first stick and the
    // input page stride.
    uint64_t bank_addr[stride_banks != 0 ? stride_banks : 1];
    uint32_t core_first_stick = 0;
    uint32_t stick_stride_bytes = 0;
    uint32_t co_read_list_arg = 0;  // co_read_listed: RT arg index of NCRISC's stick list

    explicit StickProducer(uint32_t rotation) : base_addr(get_write_ptr(cb)), stick_rotation(rotation & (tile_h - 1)) {}

    // stride_banks only: record the per-bank base addresses of this core's sticks.
    template <typename Accessor>
    FORCE_INLINE void prime(const Accessor& accessor, uint32_t first_stick, uint32_t stride_bytes) {
        if constexpr (stride_banks != 0) {
            core_first_stick = first_stick;
            stick_stride_bytes = stride_bytes;
            for (uint32_t r = 0; r < stride_banks; ++r) {
                bank_addr[r] = accessor.get_noc_addr(first_stick + r);
            }
        }
    }

    static uint32_t trid_of(uint32_t slot) { return 1 + (slot % depth); }

    // Tile-rows fully published (pushed) so far.
    uint32_t rows_pushed() const { return (next_slot - outstanding) * rows_per_slot; }

    // Open the next tile-row (opening a slot if none is open): returns its L1 base. The open
    // slot's transaction id is the NoC read state on return.
    template <typename Fill = NoPadFill>
    FORCE_INLINE uint32_t open_row(Fill& fill) {
        if constexpr (eager_publish) {
            publish_landed();
        }
        if (open_rows == 0) {
            if (outstanding == read_ahead) {
                complete_oldest(fill);
            }
            {
                MaybeDeviceZoneScope("reader_reserve");  // back-pressure from compute
                cb_reserve_back(cb, slot_pages * (outstanding + 1));
            }
            open_base = base_addr + (next_slot % depth) * slot_bytes;
            noc_async_read_set_trid(trid_of(next_slot));
            if constexpr (noc_split != 0) {
                noc_async_read_set_trid(trid_of(next_slot), 1 - noc_index);
            }
            ++next_slot;
            ++outstanding;
        }
        return open_base + open_rows * row_bytes;
    }

    FORCE_INLINE void close_row() {
        if (++open_rows == rows_per_slot) {
            open_rows = 0;  // slot sealed; it completes lazily
        }
    }

    // Padded load_block for one tile-row: read the existing sticks' data bytes, fill each
    // existing stick's W tail [data_bytes, segment) and the trailing non-existing sticks.
    // Every read of the tile-row is issued before its fills, under the slot's transaction id.
    // W-tail persistence: with one column block per core every tile-row carries the same
    // data_bytes, and a W-tail byte of a CB row is never overwritten once filled (reads stop at
    // data_bytes, whole-stick fills write the same value, compute only reads). So the band is
    // filled only on the walk's first pass through the CB ring (depth * rows_per_slot rows, in
    // step with that pass's reads so the stores still overlap them) and skipped afterwards.
    FORCE_INLINE void set_w_tail_persists() { w_tail_rows_left = depth * rows_per_slot; }
    uint32_t w_tail_rows_left = 0xFFFFFFFFu;  // tile-rows still to band-fill (all, unless persistent)

    template <typename Accessor, typename Fill>
    FORCE_INLINE void issue_row_padded(
        const Accessor& accessor, const RowSource& src, uint32_t first_col, uint32_t valid_width, Fill& fill) {
        static_assert(stride_banks == 0 && noc_split == 0 && !eager_publish, "parked levers are off when padded");
        const uint32_t l1_base = open_row(fill);
        MaybeDeviceZoneScope("reader_issue");  // stick-read issue + pad fills of one tile-row
        const uint32_t segment_bytes = valid_width * tile_col_bytes;
        const uint32_t segment_offset = first_col * tile_col_bytes;
        if (src.data_bytes != 0) {
            for (uint32_t s = 0; s < tile_h; ++s) {
                const uint32_t stick = (s + stick_rotation) & (tile_h - 1);
                if (stick >= src.valid_sticks) {
                    continue;
                }
                const uint32_t l1_dst = l1_base + stick * block_stick_bytes;
                if constexpr (pages_per_stick == 1) {
                    noc_async_read(
                        accessor.get_noc_addr(src.first_stick + stick, segment_offset), l1_dst, src.data_bytes);
                } else {
                    read_paged_segment<page_bytes, pages_per_stick>(
                        accessor, src.first_stick + stick, segment_offset, l1_dst, src.data_bytes);
                }
            }
        }
        if (w_tail_rows_left != 0) {
            // Only the existing sticks need the band: the trailing whole-stick fill below covers the
            // rest of the tile-row (band included), so a persistent CB row is complete either way.
            if (src.valid_sticks != 0 && src.data_bytes < segment_bytes) {
                fill.fill_band(
                    l1_base + src.data_bytes, block_stick_bytes, src.valid_sticks, segment_bytes - src.data_bytes);
            }
            if (w_tail_rows_left != 0xFFFFFFFFu) {
                --w_tail_rows_left;
            }
        }
        if (src.valid_sticks < tile_h) {
            // Trailing whole pad sticks (the H tail, or the whole tile-row) are one contiguous range.
            fill.fill(l1_base + src.valid_sticks * block_stick_bytes, (tile_h - src.valid_sticks) * block_stick_bytes);
        }
        noc_async_read_set_trid(trid_of(next_slot - 1));  // later rows of this slot read under its id
        close_row();
    }

    template <typename Accessor>
    FORCE_INLINE void issue_row(const Accessor& accessor, uint32_t row, uint32_t first_col, uint32_t valid_width) {
        NoPadFill no_fill;
        const uint32_t l1_base = open_row(no_fill);
        MaybeDeviceZoneScope("reader_issue");  // stick-read issue: address gen + command-buffer writes
        const uint32_t segment_bytes = valid_width * tile_col_bytes;
        const uint32_t segment_offset = first_col * tile_col_bytes;
        const uint32_t first_stick = row * tile_h;
        if constexpr (stride_banks != 0) {
            issue_row_bank_stride(first_stick, l1_base, segment_offset, segment_bytes);
        } else if constexpr (co_read_listed) {
            read_tile_row_sticks_listed<tile_h, block_stick_bytes, page_bytes, pages_per_stick>(
                accessor, first_stick, l1_base, segment_offset, segment_bytes, co_read_list_arg);
        } else {
            read_tile_row_sticks<tile_h, block_stick_bytes, page_bytes, pages_per_stick, noc_split>(
                accessor, first_stick, l1_base, segment_offset, segment_bytes, stick_rotation, 0, tile_h - co_read);
        }
        close_row();
    }

    // The tile_h stick reads of one tile-row from the primed per-bank addresses: stick at distance
    // d from core_first_stick is bank_addr[d % stride_banks] + (d / stride_banks) * stride.
    FORCE_INLINE void issue_row_bank_stride(
        uint32_t first_stick, uint32_t l1_base, uint32_t segment_offset, uint32_t segment_bytes) {
        const uint32_t row_d = first_stick - core_first_stick;
        const uint32_t row_q = row_d / stride_banks;
        const uint32_t row_r = row_d - row_q * stride_banks;
        const uint32_t row_off = row_q * stick_stride_bytes + segment_offset;
        if constexpr (bank_major) {
            constexpr uint32_t banks_per_row = stride_banks < tile_h ? stride_banks : tile_h;
            uint32_t first = stick_rotation % banks_per_row;  // in-row index of the bank's first stick
            for (uint32_t b = 0; b < banks_per_row; ++b) {
                uint32_t r = row_r + first;
                uint32_t off = row_off;
                if (r >= stride_banks) {
                    r -= stride_banks;
                    off += stick_stride_bytes;
                }
                for (uint32_t idx = first; idx < tile_h; idx += stride_banks, off += stick_stride_bytes) {
                    noc_async_read(bank_addr[r] + off, l1_base + idx * block_stick_bytes, segment_bytes);
                }
                if (++first == banks_per_row) {
                    first = 0;
                }
            }
        } else {
            uint32_t idx = stick_rotation;  // in-row stick index, visited rotated like the plain path
            uint32_t r = row_r + idx;
            uint32_t off = row_off;
            while (r >= stride_banks) {
                r -= stride_banks;
                off += stick_stride_bytes;
            }
            for (uint32_t s = 0; s < tile_h; ++s) {
                noc_async_read(bank_addr[r] + off, l1_base + idx * block_stick_bytes, segment_bytes);
                if (++idx == tile_h) {
                    idx = 0;
                    r = row_r;
                    off = row_off;
                } else if (++r == stride_banks) {
                    r = 0;
                    off += stick_stride_bytes;
                }
            }
        }
    }

    // Non-blocking: push every sealed slot, oldest first, whose reads have all landed.
    FORCE_INLINE void publish_landed() {
        while (outstanding > 0 && !(outstanding == 1 && open_rows != 0)) {
            const uint32_t trid = trid_of(next_slot - outstanding);
            if (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid)) {
                return;
            }
            if constexpr (noc_split != 0) {
                if (!ncrisc_noc_read_with_transaction_id_flushed(1 - noc_index, trid)) {
                    return;
                }
            }
            NoPadFill no_fill;
            complete_oldest(no_fill);
        }
    }

    // `fill`: the pad fills of every issued tile-row land before any slot is published.
    template <typename Fill = NoPadFill>
    FORCE_INLINE void complete_oldest(Fill& fill) {
        const uint32_t oldest = next_slot - outstanding;
        // Only the newest slot can be partial, and only at kernel end (complete_all).
        const bool partial = outstanding == 1 && open_rows != 0;
        fill.complete();
        {
            MaybeDeviceZoneScope("reader_barrier");  // the slot's stick reads landing
            noc_async_read_barrier_with_trid(trid_of(oldest));
            if constexpr (noc_split != 0) {
                noc_async_read_barrier_with_trid(trid_of(oldest), 1 - noc_index);
            }
        }
        cb_push_back(cb, partial ? open_rows * row_pages : slot_pages);
        if (partial) {
            open_rows = 0;
        }
        --outstanding;
    }

    FORCE_INLINE void complete_all() {
        NoPadFill no_fill;
        complete_all(no_fill);
    }

    template <typename Fill>
    FORCE_INLINE void complete_all(Fill& fill) {
        while (outstanding > 0) {
            complete_oldest(fill);
        }
        noc_async_read_set_trid(0);
        if constexpr (noc_split != 0) {
            noc_async_read_set_trid(0, 1 - noc_index);
        }
    }
};

// ---------------------------------------------------------------------------
// retile_l1_facewalk (op_design.md -> Regimes): `load_block` for a Layout::TILE input.
// ---------------------------------------------------------------------------
//
// A retile UNIT is `row_align` consecutive walk positions (output tile-rows) of one
// column block whose sticks come from `unit_in_rows` consecutive input tile-rows of
// ONE image:
//   * in_tile_h >= tile_h: row_align = in_tile_h / tile_h (1 where the host could not
//     align), unit_in_rows = 1 -- one input tile-row feeds row_align output tile-rows;
//   * in_tile_h <  tile_h: row_align = 1, unit_in_rows = tile_h / in_tile_h.
// The unit's input tiles (unit_in_rows x valid_width, source tile (i, c) at
// src + (i * block_width + c) * in_page_bytes) are either NoC-read whole into the
// reader-private staging ring (one page-sized read per tile) or, for a resident input
// shard, read in place from the shard. FaceWalk then re-lays their face rows into the
// stick layout of cb_input_sticks on this RISC-V: no DRAM round trip, no ROW_MAJOR
// tensor materialized. Compute and the writer never see the difference.

// Word copy of `bytes` (a multiple of 4, both ends 4-byte aligned) between L1 addresses.
template <uint32_t bytes>
FORCE_INLINE void copy_l1_words(uint32_t dst_addr, uint32_t src_addr) {
    static_assert(bytes % 4 == 0, "word copy");
    uint32_t* __restrict__ dst = reinterpret_cast<uint32_t*>(dst_addr);
    const uint32_t* __restrict__ src = reinterpret_cast<const uint32_t*>(src_addr);
#pragma GCC unroll 16
    for (uint32_t w = 0; w < bytes / 4; ++w) {
        dst[w] = src[w];
    }
}

// Face rows of an input tile [in_tile_h, 32] -> sticks of the tilize input layout.
// A tile is (in_tile_h > 16 ? 2 : 1) face-rows of two faces (left / right 16 columns),
// each face face_h = min(in_tile_h, 16) rows of 16 elements, stored face after face.
//
// Every move is one face row (16 elements) at both ends. `use_noc` selects who moves it:
//   * true:  a NoC loopback read on this core (local L1 -> local L1). The caller opens the
//            window with begin() and must complete() it before publishing the destination
//            or reusing the source; the RISC-V only issues commands.
//   * false: RISC-V word copies (loads from L1 block the core: ~8 cycles per word measured).
template <
    uint32_t in_tile_h,
    uint32_t tile_h,
    uint32_t block_width,
    uint32_t tile_col_bytes,
    uint32_t in_page_bytes,
    bool use_noc,
    uint32_t noc_trid>
struct FaceWalk {
    static_assert((in_tile_h & (in_tile_h - 1)) == 0 && in_tile_h <= 32, "in_tile_h: power of two <= 32");
    static constexpr uint32_t half_row_bytes = tile_col_bytes / 2;  // one face row: 16 elements
    static constexpr uint32_t face_h = in_tile_h < 16 ? in_tile_h : 16;
    static constexpr uint32_t face_bytes = face_h * half_row_bytes;
    static constexpr uint32_t stick_stride = block_width * tile_col_bytes;   // cb_input_sticks bytes per stick
    static constexpr uint32_t src_row_stride = block_width * in_page_bytes;  // source bytes per input tile-row

    // NoC mode: point the read command buffer at this core with a one-face-row packet, under
    // the face walk's own transaction id (any NoC read in between must be followed by begin()).
    static FORCE_INLINE void begin() {
        if constexpr (use_noc) {
            noc_async_read_set_trid(noc_trid);
            noc_async_read_one_packet_set_state(get_noc_addr(0), half_row_bytes);
        }
    }
    // NoC mode: every face row moved since begin() has landed.
    static FORCE_INLINE void complete() {
        if constexpr (use_noc) {
            noc_async_read_barrier_with_trid(noc_trid);
        }
    }
    static FORCE_INLINE void move_face_row(uint32_t dst, uint32_t src) {
        if constexpr (use_noc) {
            noc_async_read_one_packet_with_state(src, dst);
        } else {
            copy_l1_words<half_row_bytes>(dst, src);
        }
    }

    // One output tile-row at dst_row: its tile_h sticks are rows h_off .. h_off + tile_h - 1
    // of the unit's source tile-rows (row h lives in source tile-row h / in_tile_h).
    static FORCE_INLINE void tile_row(uint32_t dst_row, uint32_t src, uint32_t h_off, uint32_t valid_width) {
        for (uint32_t s = 0; s < tile_h; ++s) {
            const uint32_t h = h_off + s;
            const uint32_t rr = h % in_tile_h;
            uint32_t src_left =
                src + (h / in_tile_h) * src_row_stride + (rr / 16) * 2 * face_bytes + (rr % face_h) * half_row_bytes;
            uint32_t dst = dst_row + s * stick_stride;
            for (uint32_t c = 0; c < valid_width; ++c) {
                move_face_row(dst, src_left);
                move_face_row(dst + half_row_bytes, src_left + face_bytes);
                dst += tile_col_bytes;
                src_left += in_page_bytes;
            }
        }
    }
};

// Producer of cb_input_sticks slots filled by synchronous L1 writes (no NoC on this side):
// reserve a slot of rows_per_slot tile-rows, fill it row by row, push it; only the kernel's
// final slot may be partial (same ring-wrap invariant as StickProducer).
template <uint32_t cb, uint32_t block_width, uint32_t rows_per_slot, uint32_t row_bytes>
struct RowSlotWriter {
    static constexpr uint32_t slot_pages = rows_per_slot * block_width;
    uint32_t open_rows = 0;
    uint32_t slot_addr = 0;

    FORCE_INLINE uint32_t open_row() {
        if (open_rows == 0) {
            cb_reserve_back(cb, slot_pages);
            slot_addr = get_write_ptr(cb);
        }
        return slot_addr + open_rows * row_bytes;
    }
    // True when the row being filled is the last one before a push (its data must have landed).
    FORCE_INLINE bool row_ends_slot() const { return open_rows + 1 == rows_per_slot; }
    FORCE_INLINE void close_row() {
        if (++open_rows == rows_per_slot) {
            cb_push_back(cb, slot_pages);
            open_rows = 0;
        }
    }
    FORCE_INLINE void finish() {
        if (open_rows != 0) {
            cb_push_back(cb, open_rows * block_width);
            open_rows = 0;
        }
    }
};

// One retile unit: its first output tile-row and column block, plus where its sticks start
// in the input tile grid (first input tile-row, row offset inside it).
struct RetileUnit {
    uint32_t row, first_col, valid_width, first_in_row, h_off;
};

// Output tile-row `row` (global over R) -> (first input tile-row, row offset in it), per image:
// the input's tile-rows per image (in_rows_per_image) count its own H padding.
FORCE_INLINE void retile_source_of(
    uint32_t row,
    uint32_t tile_h,
    uint32_t in_tile_h,
    uint32_t out_rows_per_image,
    uint32_t in_rows_per_image,
    uint32_t& first_in_row,
    uint32_t& h_off) {
    const uint32_t image = row / out_rows_per_image;
    const uint32_t h0 = (row - image * out_rows_per_image) * tile_h;
    first_in_row = image * in_rows_per_image + h0 / in_tile_h;
    h_off = h0 % in_tile_h;
}

// store_block for `num_rows` consecutive walk positions (one CB quantum): wait
// num_rows * block_width pages, valid_width tile-page writes per tile-row in
// flight, one flush (L1 source reads done), pop. `walk` is advanced past them.
// The writes of one tile-row start at column `col_rotation % valid_width` and
// wrap (the write twin of the reader's stick rotation): when every Tensix core
// holds the same few tile-rows at once, concurrent first writes would otherwise
// all land on the banks of column 0. `noc_split` (parked, 0 = off): every
// noc_split-th write of a tile-row goes out on the other NoC (DM_DYNAMIC_NOC).
template <
    uint32_t cb_output_tiles,
    uint32_t block_width,
    uint32_t out_tile_bytes,
    uint32_t noc_split = 0,
    typename Accessor,
    typename Walk>
FORCE_INLINE void store_rows(
    const Accessor& accessor, Walk& walk, uint32_t tiles_per_row, uint32_t num_rows, uint32_t col_rotation) {
    const uint32_t pages = num_rows * block_width;
    {
        MaybeDeviceZoneScope("writer_wait");  // starved on compute
        cb_wait_front(cb_output_tiles, pages);
    }
    {
        MaybeDeviceZoneScope("writer_issue");  // tile-write issue: address gen + command-buffer writes
        uint32_t l1_row_addr = get_read_ptr(cb_output_tiles);
        for (uint32_t j = 0; j < num_rows; ++j, walk.advance()) {
            const uint32_t row_tile_idx = walk.row() * tiles_per_row + walk.first_col();
            const uint32_t valid_width = walk.valid_width();
            uint32_t t = col_rotation % valid_width;
            for (uint32_t n = 0; n < valid_width; ++n) {
                const uint8_t noc = (noc_split != 0 && (n % noc_split) == noc_split - 1) ? 1 - noc_index : noc_index;
                noc_async_write(
                    l1_row_addr + t * out_tile_bytes,
                    accessor.get_noc_addr(row_tile_idx + t, 0, noc),
                    out_tile_bytes,
                    noc);
                if (++t == valid_width) {
                    t = 0;
                }
            }
            l1_row_addr += block_width * out_tile_bytes;
        }
    }
    {
        MaybeDeviceZoneScope("writer_flush");  // the quantum's L1 source reads done (not landed)
        noc_async_writes_flushed();
        if constexpr (noc_split != 0) {
            noc_async_writes_flushed(1 - noc_index);
        }
    }
    cb_pop_front(cb_output_tiles, pages);
}

// store_block with a write-ahead window (the write twin of StickProducer's read-ahead): up to
// `write_ahead` CB quanta of tile-page writes in flight, each quantum tagged with its own NoC
// transaction id; the oldest quantum is flushed (its L1 source reads done) and popped only when
// the window is full or at the end. write_ahead == 1 is store_rows' behavior. Every quantum but
// the kernel's last is the nominal rows_per_slot * block_width pages, so quantum k sits at
// base + (k % depth) * slot_bytes in the CB ring. Requires write_ahead <= depth <= 15.
template <
    uint32_t cb,
    uint32_t block_width,
    uint32_t out_tile_bytes,
    uint32_t depth,
    uint32_t write_ahead,
    uint32_t rows_per_slot>
struct TileStorer {
    static_assert(write_ahead >= 1 && write_ahead <= depth && depth <= 15, "one write trid per CB slot");
    static constexpr uint32_t slot_pages = rows_per_slot * block_width;
    static constexpr uint32_t slot_bytes = slot_pages * out_tile_bytes;
    uint32_t base_addr;
    uint32_t next_slot = 0;
    uint32_t outstanding = 0;
    uint32_t last_pages = slot_pages;  // pages of the newest slot (only the final one may be short)

    TileStorer() : base_addr(get_read_ptr(cb)) {}
    static uint32_t trid_of(uint32_t slot) { return 1 + (slot % depth); }

    template <typename Accessor, typename Walk>
    FORCE_INLINE void store(
        const Accessor& accessor, Walk& walk, uint32_t tiles_per_row, uint32_t num_rows, uint32_t col_rotation) {
        if (outstanding == write_ahead) {
            complete_oldest();
        }
        const uint32_t pages = num_rows * block_width;
        {
            MaybeDeviceZoneScope("writer_wait");  // starved on compute
            cb_wait_front(cb, outstanding * slot_pages + pages);
        }
        MaybeDeviceZoneScope("writer_issue");
        const uint32_t trid = trid_of(next_slot);
        uint32_t l1_row_addr = base_addr + (next_slot % depth) * slot_bytes;
        for (uint32_t j = 0; j < num_rows; ++j, walk.advance()) {
            const uint32_t row_tile_idx = walk.row() * tiles_per_row + walk.first_col();
            const uint32_t valid_width = walk.valid_width();
            uint32_t t = col_rotation % valid_width;
            for (uint32_t n = 0; n < valid_width; ++n) {
                noc_async_write_one_packet_with_trid(
                    l1_row_addr + t * out_tile_bytes, accessor.get_noc_addr(row_tile_idx + t), out_tile_bytes, trid);
                if (++t == valid_width) {
                    t = 0;
                }
            }
            l1_row_addr += block_width * out_tile_bytes;
        }
        last_pages = pages;
        ++next_slot;
        ++outstanding;
    }

    FORCE_INLINE void complete_oldest() {
        const uint32_t oldest = next_slot - outstanding;
        {
            MaybeDeviceZoneScope("writer_flush");
            noc_async_write_flushed_with_trid(trid_of(oldest));
        }
        cb_pop_front(cb, outstanding == 1 ? last_pages : slot_pages);
        --outstanding;
    }

    FORCE_INLINE void complete_all() {
        while (outstanding > 0) {
            complete_oldest();
        }
    }
};

}  // namespace tilize_dataflow
