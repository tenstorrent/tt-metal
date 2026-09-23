// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// pixel_unshuffle, channels_last: NCHW ROW_MAJOR input -> NHWC [N, Ho, Wo, Cp] ROW_MAJOR,
// HEIGHT_SHARDED in L1, written into this core's own shard.
//
// This core owns output pixels [pix0, pix0 + npix) in flattened (n, ho, wo) order; that is
// exactly its shard, so output stick j of the shard is pixel pix0 + j. The two dataflow
// RISCs run this same source; each image row touched by the range is split between them
// at a DRAM-aligned column, and each RISC reads only the r*C input half-rows its columns
// need (page = input row n*C*H + c*H + ho*r + dy, a byte span of it), then gathers into
// the shard with plain 32-bit L1 stores. There are no NOC writes.
//
// Fast path (2-byte elements, r even, CHANNEL_MAJOR): pixel v takes, from input row
// (c, dy), the r elements at columns [v*r, v*r + r) - r/2 naturally aligned 32-bit words -
// and puts them at channels [c*r*r + dy*r, +r): also r/2 aligned words. The gather is a
// pure word copy, one load + one store per word, no shifts. Everything else (odd r,
// 4-byte elements, SPATIAL_MAJOR) takes the element-wise path.
//
// Reads run one image row ahead of the gather (depth 2), with plain barriers only: the
// barrier at the top of an iteration waits for exactly the reads issued in the previous
// one. No transaction ids (see pixel_unshuffle_nchw.cpp for why).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t cgcd(uint32_t a, uint32_t b) { return b == 0 ? a : cgcd(b, a % b); }

struct Item {
    uint32_t n;
    uint32_t ho;
    uint32_t va;     // first output column of this RISC's span
    uint32_t vb;     // one past the last
    uint32_t ca_al;  // first input column read (aligned down)
    uint32_t cb_al;  // one past the last input column read (aligned up, <= W)
    uint32_t lpix0;  // shard-local index of pixel (n, ho, va)
};

}  // namespace

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr uint32_t C = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t r = get_compile_time_arg_val(3);
    constexpr uint32_t Ho = get_compile_time_arg_val(4);
    constexpr uint32_t Wo = get_compile_time_arg_val(5);
    constexpr uint32_t datum = get_compile_time_arg_val(6);
    constexpr uint32_t Cp = get_compile_time_arg_val(7);
    constexpr uint32_t out_row_nbytes = get_compile_time_arg_val(8);
    constexpr uint32_t aligned_row_nbytes_in = get_compile_time_arg_val(9);
    constexpr uint32_t cb_in = get_compile_time_arg_val(10);
    constexpr uint32_t cb_out = get_compile_time_arg_val(11);
    constexpr uint32_t channel_order = get_compile_time_arg_val(12);  // 0=CHANNEL_MAJOR, 1=SPATIAL_MAJOR
    constexpr uint32_t depth = get_compile_time_arg_val(13);
    constexpr uint32_t risc = get_compile_time_arg_val(14);  // 0: first half of each row, 1: second
    constexpr auto src_args = TensorAccessorArgs<15, 0>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t pix0 = get_arg_val<uint32_t>(1);
    const uint32_t npix = get_arg_val<uint32_t>(2);
    if (npix == 0) {
        return;
    }

    constexpr uint32_t SPATIAL_MAJOR = 1;
    constexpr uint32_t HoWo = Ho * Wo;
    constexpr uint32_t CH = C * H;
    constexpr uint32_t C_out = C * r * r;
    constexpr uint32_t rows_per_item = C * r;
    constexpr uint32_t slot_nbytes = rows_per_item * aligned_row_nbytes_in;

    // DRAM reads want 32 B-aligned source offsets and matching L1 alignment, so a RISC's
    // input span is aligned to ALIGN_ELEMS input columns. A split between the two RISCs at
    // output column s puts the input boundary at s*r, so s is a multiple of SPLIT_GRAN.
    constexpr uint32_t ALIGN_ELEMS = 32 / datum;
    constexpr uint32_t SPLIT_GRAN = ALIGN_ELEMS / cgcd(ALIGN_ELEMS, r);
    constexpr bool fast = (datum == 2) && ((r % 2) == 0) && (channel_order != SPATIAL_MAJOR);

    const auto s_in = TensorAccessor(src_args, src_addr);
    const uint32_t in_base = get_write_ptr(cb_in);
    const uint32_t out_base = get_write_ptr(cb_out);
    const uint32_t end = pix0 + npix;

    // Walk the pixel range one image row at a time and hand this RISC its half of the row.
    // Returns false when the range is exhausted. `cursor` is the global pixel index.
    auto advance = [&](uint32_t& cursor, Item& it) -> bool {
        while (cursor < end) {
            const uint32_t n = cursor / HoWo;
            const uint32_t rem = cursor - n * HoWo;
            const uint32_t ho = rem / Wo;
            const uint32_t wa = rem - ho * Wo;
            const uint32_t row_len = (end - cursor < Wo - wa) ? (end - cursor) : (Wo - wa);
            const uint32_t wb = wa + row_len;
            const uint32_t row_first = cursor;
            cursor += row_len;

            // Split so both halves start on a DRAM-aligned input column.
            const uint32_t half = ((row_len / 2) / SPLIT_GRAN) * SPLIT_GRAN;
            uint32_t va;
            uint32_t vb;
            if (half == 0) {  // too short to split: RISC 0 takes it all
                if (risc != 0) {
                    continue;
                }
                va = wa;
                vb = wb;
            } else if (risc == 0) {
                va = wa;
                vb = wa + half;
            } else {
                va = wa + half;
                vb = wb;
            }
            it.n = n;
            it.ho = ho;
            it.va = va;
            it.vb = vb;
            it.ca_al = (va * r) & ~(ALIGN_ELEMS - 1);
            const uint32_t cb = vb * r;
            const uint32_t cb_up = (cb + ALIGN_ELEMS - 1) & ~(ALIGN_ELEMS - 1);
            it.cb_al = cb_up < W ? cb_up : W;
            it.lpix0 = row_first - pix0 + (va - wa);
            return true;
        }
        return false;
    };

    auto issue_reads = [&](const Item& it, uint32_t slot) {
        const uint32_t nbytes = (it.cb_al - it.ca_al) * datum;
        const uint32_t src_off = it.ca_al * datum;
        uint32_t dst = in_base + slot * slot_nbytes;
        const uint32_t page_base = it.n * CH + it.ho * r;  // + c*H + dy
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t dy = 0; dy < r; dy++) {
                noc_async_read(s_in.get_noc_addr(page_base + c * H + dy, src_off), dst, nbytes);
                dst += aligned_row_nbytes_in;
            }
        }
    };

    auto gather = [&](const Item& it, uint32_t slot) {
        const uint32_t slot_base = in_base + slot * slot_nbytes;
        uint32_t dst = out_base + it.lpix0 * out_row_nbytes;

        if constexpr (fast) {
            // Word gather: pixel v, row (c, dy) -> r/2 words at src word (v*r - ca_al)/2,
            // landing at dst word (c*r*r + dy*r)/2. Two pixels per iteration keep 2*C*r*r/2
            // loads in flight before the first store.
            constexpr uint32_t wpr = r / 2;      // words per (c, dy) per pixel
            constexpr uint32_t wpp = C_out / 2;  // data words per pixel
            constexpr uint32_t wpad = Cp / 2;    // total words per pixel
            const uint32_t* rows[rows_per_item];
            for (uint32_t k = 0; k < rows_per_item; k++) {
                rows[k] = (const uint32_t*)(slot_base + k * aligned_row_nbytes_in);
            }
            uint32_t v = it.va;
            uint32_t sw = (it.va * r - it.ca_al) / 2;  // source word index of pixel v
            for (; v + 1 < it.vb; v += 2) {
                uint32_t w0[wpp];
                uint32_t w1[wpp];
#pragma GCC unroll 32
                for (uint32_t k = 0; k < rows_per_item; k++) {
#pragma GCC unroll 8
                    for (uint32_t t = 0; t < wpr; t++) {
                        w0[k * wpr + t] = rows[k][sw + t];
                        w1[k * wpr + t] = rows[k][sw + wpr + t];  // pixel v+1 starts r/2 words on
                    }
                }
                tt_l1_ptr uint32_t* d0 = (tt_l1_ptr uint32_t*)dst;
                tt_l1_ptr uint32_t* d1 = (tt_l1_ptr uint32_t*)(dst + out_row_nbytes);
#pragma GCC unroll 32
                for (uint32_t q = 0; q < wpp; q++) {
                    d0[q] = w0[q];
                    d1[q] = w1[q];
                }
#pragma GCC unroll 32
                for (uint32_t q = wpp; q < wpad; q++) {
                    d0[q] = 0;
                    d1[q] = 0;
                }
                dst += 2 * out_row_nbytes;
                sw += 2 * wpr;
            }
            if (v < it.vb) {  // odd tail
                tt_l1_ptr uint32_t* d0 = (tt_l1_ptr uint32_t*)dst;
#pragma GCC unroll 32
                for (uint32_t k = 0; k < rows_per_item; k++) {
#pragma GCC unroll 8
                    for (uint32_t t = 0; t < wpr; t++) {
                        d0[k * wpr + t] = rows[k][sw + t];
                    }
                }
#pragma GCC unroll 32
                for (uint32_t q = wpp; q < wpad; q++) {
                    d0[q] = 0;
                }
            }
        } else if constexpr (datum == 2) {
            // Element path, 16-bit: c_out(c, dy, dx) per channel_order.
            const uint16_t* rows[rows_per_item];
            for (uint32_t k = 0; k < rows_per_item; k++) {
                rows[k] = (const uint16_t*)(slot_base + k * aligned_row_nbytes_in);
            }
            for (uint32_t v = it.va; v < it.vb; v++) {
                tt_l1_ptr uint16_t* d = (tt_l1_ptr uint16_t*)dst;
                const uint32_t se = v * r - it.ca_al;
                for (uint32_t c = 0; c < C; c++) {
                    for (uint32_t dy = 0; dy < r; dy++) {
                        const uint16_t* srow = rows[c * r + dy] + se;
                        for (uint32_t dx = 0; dx < r; dx++) {
                            const uint32_t co = (channel_order == SPATIAL_MAJOR) ? (dy * (r * C) + dx * C + c)
                                                                                 : (c * r * r + dy * r + dx);
                            d[co] = srow[dx];
                        }
                    }
                }
                for (uint32_t co = C_out; co < Cp; co++) {
                    d[co] = 0;
                }
                dst += out_row_nbytes;
            }
        } else {
            // Element path, 32-bit elements.
            const uint32_t* rows[rows_per_item];
            for (uint32_t k = 0; k < rows_per_item; k++) {
                rows[k] = (const uint32_t*)(slot_base + k * aligned_row_nbytes_in);
            }
            for (uint32_t v = it.va; v < it.vb; v++) {
                tt_l1_ptr uint32_t* d = (tt_l1_ptr uint32_t*)dst;
                const uint32_t se = v * r - it.ca_al;
                for (uint32_t c = 0; c < C; c++) {
                    for (uint32_t dy = 0; dy < r; dy++) {
                        const uint32_t* srow = rows[c * r + dy] + se;
                        for (uint32_t dx = 0; dx < r; dx++) {
                            const uint32_t co = (channel_order == SPATIAL_MAJOR) ? (dy * (r * C) + dx * C + c)
                                                                                 : (c * r * r + dy * r + dx);
                            d[co] = srow[dx];
                        }
                    }
                }
                for (uint32_t co = C_out; co < Cp; co++) {
                    d[co] = 0;
                }
                dst += out_row_nbytes;
            }
        }
    };

    uint32_t cursor = pix0;
    Item cur;
    Item nxt;
    if (!advance(cursor, cur)) {
        return;
    }
    uint32_t slot = 0;
    issue_reads(cur, slot);
    while (true) {
        const bool has_next = advance(cursor, nxt);
        noc_async_read_barrier();  // only cur's reads are outstanding
        asm volatile("" ::: "memory");
        if (has_next) {
            issue_reads(nxt, slot ^ 1);  // overlaps the gather below
        }
        gather(cur, slot);
        if (!has_next) {
            break;
        }
        cur = nxt;
        slot ^= 1;
    }
    (void)depth;  // pipeline depth is fixed at 2 by the slot^1 scheme; kept for the factory's CB sizing
}
