// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// riscv_transpose bake-off writer (BRISC) of nhwc_rm_transpose: out[n, c, h, w] = x[n, h, w, c].
// Same units, cursor and partial-row writes as kernels/g6_permute_nhwc_rm_writer.cpp; what changes is
// who transposes and how:
//   solo (n_helpers == 0): this RISC transposes every unit itself with impl `impl` (rvt_transpose.hpp)
//     into one rows buffer, then writes; impl IMPL_BASE with batch 1 is the op's writer.
//   shared (n_helpers > 0): the transpose of each unit is split by row items (rvt::transpose_rows) over
//     the helpers (TRISC 0..2 and/or NCRISC, each posting a done counter per block) and, when
//     own_lo < own_hi, this RISC. The rows buffer has rows_depth generations: helpers fill block b + 1
//     while this RISC writes block b; the free counter tells them which generations have left L1.
// Raw dataflow and raw RISC-V transpose: kernel_lib has no helper that transposes halfwords between
// strided L1 rows, and no multi-producer CB primitive (see rvt_sync.hpp).

#include "api/dataflow/dataflow_api.h"
#include "g6_permute_common.hpp"
#include "rvt_transpose.hpp"
#include "rvt_sync.hpp"

constexpr uint32_t cb_input_slots = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_rows = get_compile_time_arg_val(1);
constexpr uint32_t block_units = get_compile_time_arg_val(2);
constexpr uint32_t tensor_h = get_compile_time_arg_val(3);
constexpr uint32_t tensor_c = get_compile_time_arg_val(4);
constexpr uint32_t units_per_row = get_compile_time_arg_val(5);
constexpr uint32_t block_w = get_compile_time_arg_val(6);
constexpr uint32_t tail_w = get_compile_time_arg_val(7);
constexpr uint32_t slot_bytes = get_compile_time_arg_val(8);
constexpr uint32_t row_bytes = get_compile_time_arg_val(9);
constexpr uint32_t out_page_bytes = get_compile_time_arg_val(10);
constexpr uint32_t impl = get_compile_time_arg_val(11);
constexpr uint32_t batch = get_compile_time_arg_val(12);
constexpr uint32_t input_depth = get_compile_time_arg_val(13);
constexpr uint32_t rows_depth = get_compile_time_arg_val(14);
constexpr uint32_t cb_free = get_compile_time_arg_val(15);
constexpr uint32_t cb_done0 = get_compile_time_arg_val(16);
constexpr uint32_t n_helpers = get_compile_time_arg_val(17);
constexpr uint32_t own_lo = get_compile_time_arg_val(18);
constexpr uint32_t own_hi = get_compile_time_arg_val(19);
constexpr auto output_args = TensorAccessorArgs<20>();
#ifdef RVT_BRISC_READ_DEN
// S1 read split: this RISC also reads the last width * NUM / DEN pages of every unit on its own NoC
// (the reader reads the rest), input_depth blocks ahead of its writes, and posts RVT_CB_BRD per block.
constexpr auto input_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
constexpr uint32_t in_page_bytes = RVT_IN_PAGE_BYTES;
constexpr uint32_t brisc_read_num = RVT_BRISC_READ_NUM;
constexpr uint32_t brisc_read_den = RVT_BRISC_READ_DEN;
#ifdef RVT_SPLIT0_DEN
constexpr uint32_t brisc_read0_num = RVT_SPLIT0_NUM;
constexpr uint32_t brisc_read0_den = RVT_SPLIT0_DEN;
#else
constexpr uint32_t brisc_read0_num = brisc_read_num;
constexpr uint32_t brisc_read0_den = brisc_read_den;
#endif
#endif

constexpr uint32_t value_bytes = sizeof(uint16_t);
constexpr uint32_t unit_row_bytes = tensor_c * row_bytes;
constexpr uint32_t block_slot_bytes = block_units * block_w * slot_bytes;
constexpr uint32_t block_row_bytes = block_units * unit_row_bytes;
constexpr uint32_t PB = 16;
// Helpers whose done counters gate a block: bit t = TRISC t (counter cb_done0 + t), bit 3 = NCRISC.
#ifdef RVT_HELPER_MASK
constexpr uint32_t helper_mask = RVT_HELPER_MASK;
#else
constexpr uint32_t helper_mask = 7;
#endif

static_assert(tail_w >= 1 && tail_w <= block_w && row_bytes >= block_w * value_bytes);
static_assert(out_page_bytes == ((units_per_row - 1) * block_w + tail_w) * value_bytes, "output page is one row of W");
static_assert(units_per_row == 1 || (block_w * value_bytes) % 16 == 0, "partial rows start 16 B-aligned");
static_assert(n_helpers > 0 || rows_depth == 1);

struct OutCursor {
    uint32_t wb, h, page;

    bool is_tail() const { return wb == units_per_row - 1; }

    void advance() {
        if (++wb == units_per_row) {
            wb = 0;
            ++page;
            if (++h == tensor_h) {
                h = 0;
                page += (tensor_c - 1) * tensor_h;
            }
        }
    }
};

// Solo: every row of every unit of the block, with impl `impl`.
FORCE_INLINE void transpose_block_solo(uint32_t src, uint32_t dst, OutCursor cursor, uint32_t extent) {
    asm volatile("" ::: "memory");
    for (uint32_t u = 0; u < extent; ++u) {
        if (tail_w != block_w && cursor.is_tail()) {
            if constexpr (impl == rvt::IMPL_BLKV) {
                rvt::transpose_rows<tail_w, tensor_c, slot_bytes, row_bytes, PB>(
                    src, dst, 0, rvt::row_items<tensor_c>());
            } else {
                rvt::transpose_range<tail_w, tensor_c, slot_bytes, row_bytes, impl, batch>(src, dst, 0, tail_w);
            }
            src += tail_w * slot_bytes;
        } else {
            if constexpr (impl == rvt::IMPL_BLKV) {
                rvt::transpose_rows<block_w, tensor_c, slot_bytes, row_bytes, PB>(
                    src, dst, 0, rvt::row_items<tensor_c>());
            } else {
                rvt::transpose_range<block_w, tensor_c, slot_bytes, row_bytes, impl, batch>(src, dst, 0, block_w);
            }
            src += block_w * slot_bytes;
        }
        dst += unit_row_bytes;
        cursor.advance();
    }
    asm volatile("" ::: "memory");
}

// Shared: this RISC's row items [own_lo, own_hi) of every unit of the block.
FORCE_INLINE void transpose_block_own(uint32_t src, uint32_t dst, OutCursor cursor, uint32_t extent) {
    asm volatile("" ::: "memory");
    for (uint32_t u = 0; u < extent; ++u) {
        if (tail_w != block_w && cursor.is_tail()) {
            rvt::transpose_rows<tail_w, tensor_c, slot_bytes, row_bytes, PB>(src, dst, own_lo, own_hi);
            src += tail_w * slot_bytes;
        } else {
            rvt::transpose_rows<block_w, tensor_c, slot_bytes, row_bytes, PB>(src, dst, own_lo, own_hi);
            src += block_w * slot_bytes;
        }
        dst += unit_row_bytes;
        cursor.advance();
    }
    asm volatile("" ::: "memory");
}

#ifdef RVT_BRISC_READ_DEN
struct ReadCursor {
    uint32_t page, wb, remaining, block;
};

// This RISC's pages of the next block (block rc.block) into its slots; returns false when none is left.
template <typename Accessor>
FORCE_INLINE bool issue_share(const Accessor& input, ReadCursor& rc, uint32_t slots_base) {
    if (rc.remaining == 0) {
        return false;
    }
    const uint32_t extent = rc.remaining < block_units ? rc.remaining : block_units;
    uint32_t slot = slots_base + (rc.block % input_depth) * block_slot_bytes;
    for (uint32_t u = 0; u < extent; ++u) {
        const uint32_t width = (tail_w != block_w && rc.wb == units_per_row - 1) ? tail_w : block_w;
        const uint32_t own =
            rc.block == 0 ? width * brisc_read0_num / brisc_read0_den : width * brisc_read_num / brisc_read_den;
        for (uint32_t p = width - own; p < width; ++p) {
            noc_async_read<in_page_bytes>(input.get_noc_addr(rc.page + p), slot + p * slot_bytes, in_page_bytes);
        }
        rc.page += width;
        slot += width * slot_bytes;
        if (++rc.wb == units_per_row) {
            rc.wb = 0;
        }
    }
    rc.remaining -= extent;
    return true;
}

FORCE_INLINE void post_share(ReadCursor& rc) {
    noc_async_read_barrier();
    rvt::ctr_post(RVT_CB_BRD, ++rc.block);
}
#endif

template <typename Accessor>
FORCE_INLINE void write_block(const Accessor& output, uint32_t rows, OutCursor& cursor, uint32_t extent) {
    uint32_t row = rows;
    for (uint32_t u = 0; u < extent; ++u) {
        const uint32_t bytes = (cursor.is_tail() ? tail_w : block_w) * value_bytes;
        const uint32_t offset = cursor.wb * block_w * value_bytes;
        for (uint32_t c = 0; c < tensor_c; ++c) {
            noc_async_write<block_w * value_bytes>(row, output.get_noc_addr(cursor.page + c * tensor_h, offset), bytes);
            row += row_bytes;
        }
        cursor.advance();
    }
}

template <bool zoned, typename Accessor, typename Reads>
FORCE_INLINE void handle_block(
    const Accessor& output,
    Reads& reads,
    uint32_t slots_base,
    uint32_t rows_base,
    OutCursor& cursor,
    uint32_t extent,
    uint32_t b) {
    const uint32_t src = slots_base + (b % input_depth) * block_slot_bytes;
    const uint32_t rows = rows_base + (b % rows_depth) * block_row_bytes;
    if constexpr (n_helpers == 0) {
        G6_ZONED_STEP(zoned, "RVT-WR-WAIT", cb_wait_front(cb_input_slots, block_units));
        // The rows are the previous block's write source; the flush also drops stale cache lines.
        G6_ZONED_STEP(zoned, "RVT-WR-FLUSH", noc_async_writes_flushed());
        rvt::l1_invalidate();
        G6_ZONED_STEP(zoned, "RVT-WR-XPOSE", transpose_block_solo(src, rows, cursor, extent));
    } else {
        if constexpr (own_lo < own_hi) {
            G6_ZONED_STEP(zoned, "RVT-WR-WAIT", cb_wait_front(cb_input_slots, block_units));
            // Generation b % rows_depth last held block b - rows_depth, flushed at block b - 1 at the latest.
            rvt::l1_invalidate();
            G6_ZONED_STEP(zoned, "RVT-WR-XPOSE", transpose_block_own(src, rows, cursor, extent));
        }
        {
            G6_ZONED_STEP(
                zoned, "RVT-WR-WAIT-HELPERS", for (uint32_t t = 0; t < 4; ++t) {
                    if ((helper_mask >> t) & 1) {
                        rvt::ctr_wait(cb_done0 + t, b + 1);
                    }
                });
        }
        if constexpr (own_lo >= own_hi) {
            cb_wait_front(cb_input_slots, block_units);
        }
    }
    cb_pop_front(cb_input_slots, block_units);
    if constexpr (n_helpers > 0 && rows_depth > 1) {
        if (b >= 1) {
            // Blocks 0 .. b - 1 have left L1: their rows generations are free again.
            G6_ZONED_STEP(zoned, "RVT-WR-FLUSH", noc_async_writes_flushed());
            rvt::ctr_post(cb_free, b);
        }
    }
#ifdef RVT_BRISC_READ_DEN
    // Block b's slots are free again: this RISC's share of block b + input_depth flies with the writes.
    const bool issued = issue_share(reads.input, reads.rc, slots_base);
#endif
    G6_ZONED_STEP(zoned, "RVT-WR-ISSUE", write_block(output, rows, cursor, extent));
#ifdef RVT_BRISC_READ_DEN
    if (issued) {
        G6_ZONED_STEP(zoned, "RVT-WR-READ-BARRIER", post_share(reads.rc));
    }
#endif
    if constexpr (n_helpers > 0 && rows_depth == 1) {
        G6_ZONED_STEP(zoned, "RVT-WR-FLUSH", noc_async_writes_flushed());
        rvt::ctr_post(cb_free, b + 1);
    }
}

#ifdef RVT_STREAM
// Streamed writes: each helper posts one count per row item it finishes (TRISC t: items
// [RVT_LO<t>, RVT_HI<t>) of every unit, in order), and the rows of an item go out as soon as it is done,
// so the writes of a unit overlap its transpose and only the last item's writes trail the last unit.
constexpr uint32_t s_lo[3] = {RVT_LO0, RVT_LO1, RVT_LO2};
constexpr uint32_t s_n[3] = {RVT_HI0 - RVT_LO0, RVT_HI1 - RVT_LO1, RVT_HI2 - RVT_LO2};
constexpr uint32_t s_max = s_n[0] > s_n[1] ? (s_n[0] > s_n[2] ? s_n[0] : s_n[2]) : (s_n[1] > s_n[2] ? s_n[1] : s_n[2]);

template <typename Accessor>
FORCE_INLINE void write_item(const Accessor& output, uint32_t unit_rows, const OutCursor& cursor, uint32_t m) {
    const uint32_t bytes = (cursor.is_tail() ? tail_w : block_w) * value_bytes;
    const uint32_t offset = cursor.wb * block_w * value_bytes;
    const uint32_t c_hi = 2 * m + 2 < tensor_c ? 2 * m + 2 : tensor_c;
    for (uint32_t c = 2 * m; c < c_hi; ++c) {
        noc_async_write<block_w * value_bytes>(
            unit_rows + c * row_bytes, output.get_noc_addr(cursor.page + c * tensor_h, offset), bytes);
    }
}

template <typename Accessor>
FORCE_INLINE void stream_block(
    const Accessor& output, uint32_t rows, OutCursor& cursor, uint32_t extent, uint32_t& units_done) {
    for (uint32_t u = 0; u < extent; ++u) {
        const uint32_t unit_rows = rows + u * unit_row_bytes;
        const uint32_t ug = units_done + u;
        for (uint32_t i = 0; i < s_max; ++i) {
            for (uint32_t t = 0; t < 3; ++t) {
                if (i < s_n[t]) {
                    rvt::ctr_wait16(cb_done0 + t, ug * s_n[t] + i + 1);
                    write_item(output, unit_rows, cursor, s_lo[t] + i);
                }
            }
        }
        cursor.advance();
    }
    units_done += extent;
}

template <bool zoned, typename Accessor, typename Reads>
FORCE_INLINE void handle_block_stream(
    const Accessor& output,
    Reads& reads,
    uint32_t slots_base,
    uint32_t rows_base,
    OutCursor& cursor,
    uint32_t extent,
    uint32_t b,
    uint32_t& units_done) {
    if (b >= 1) {
        // Blocks 0 .. b - 1 have left L1: their rows generations are free again.
        G6_ZONED_STEP(zoned, "RVT-WR-FLUSH", noc_async_writes_flushed());
        rvt::ctr_post(cb_free, b);
    }
    G6_ZONED_STEP(
        zoned,
        "RVT-WR-STREAM",
        stream_block(output, rows_base + (b % rows_depth) * block_row_bytes, cursor, extent, units_done));
    cb_wait_front(cb_input_slots, block_units);
    cb_pop_front(cb_input_slots, block_units);
#ifdef RVT_BRISC_READ_DEN
    if (issue_share(reads.input, reads.rc, slots_base)) {
        G6_ZONED_STEP(zoned, "RVT-WR-READ-BARRIER", post_share(reads.rc));
    }
#endif
}
#endif

void kernel_main() {
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    uint32_t remaining = get_arg_val<uint32_t>(0);
    OutCursor cursor{get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3)};

    const auto output = TensorAccessor(output_args, output_addr, out_page_bytes);
    const uint32_t rows_base = get_write_ptr(cb_output_rows);
    const uint32_t slots_base = get_read_ptr(cb_input_slots);

#ifdef RVT_BRISC_READ_DEN
    struct {
        decltype(TensorAccessor(input_args, 0u, 0u)) input;
        ReadCursor rc;
    } reads{
        TensorAccessor(input_args, get_common_arg_val<uint32_t>(1), in_page_bytes),
        {get_arg_val<uint32_t>(4), cursor.wb, remaining, 0}};
    {
        DeviceZoneScopedN("RVT-WR-READ-PROLOGUE");
        for (uint32_t s = 0; s < input_depth && issue_share(reads.input, reads.rc, slots_base); ++s) {
            post_share(reads.rc);
        }
    }
#else
    struct {
    } reads;
#endif

    uint32_t b = 0;
#ifdef RVT_STREAM
    uint32_t units_done = 0;
    for (; b < g6_permute::ZONED_BLOCKS && remaining > 0; ++b) {
        DeviceZoneScopedN("RVT-WR-BLOCK");
        const uint32_t extent = remaining < block_units ? remaining : block_units;
        handle_block_stream<true>(output, reads, slots_base, rows_base, cursor, extent, b, units_done);
        remaining -= extent;
    }
    for (; remaining > 0; ++b) {
        const uint32_t extent = remaining < block_units ? remaining : block_units;
        handle_block_stream<false>(output, reads, slots_base, rows_base, cursor, extent, b, units_done);
        remaining -= extent;
    }
#else
    for (; b < g6_permute::ZONED_BLOCKS && remaining > 0; ++b) {
        DeviceZoneScopedN("RVT-WR-BLOCK");
        const uint32_t extent = remaining < block_units ? remaining : block_units;
        handle_block<true>(output, reads, slots_base, rows_base, cursor, extent, b);
        remaining -= extent;
    }
    for (; remaining > 0; ++b) {
        const uint32_t extent = remaining < block_units ? remaining : block_units;
        handle_block<false>(output, reads, slots_base, rows_base, cursor, extent, b);
        remaining -= extent;
    }
#endif

    {
        DeviceZoneScopedN("RVT-WR-FINAL-BARRIER");
        noc_async_write_barrier();
    }
}
