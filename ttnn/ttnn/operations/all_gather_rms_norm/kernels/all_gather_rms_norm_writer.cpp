// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm worker writer (BRISC / NoC1). Block operations (op_design.md -> Block schedule):
//   publish_records_block  column 0 of each stat tile of cb_partial -> a 128 B fp32 record, written into BOTH hubs'
//                          stats tables at slot(p) + ((tr * k + m) * nstats + s) * 128; one barrier; one
//                          records_local increment per hub
//   store_own_block        the normalized block (cb_out_block, kept resident: the ports pull it from here) -> own
//                          output pages (aligned) / own staging slot (R_U); then one own_ready increment per port
//   assemble_block (R_U)   after every port announced its arrivals landed: the worker's output tiles, each built
//                          from the staging slots by element-shifted copies (segments split at 16-column faces on
//                          both sides), written to the output
//
// Raw dataflow (documented): record extraction and the element-shifted assembly are word / datum L1 copies no
// helper expresses (op_design.md "Helpers considered and rejected").

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "agrn_worker_args.hpp"

using namespace agrn;

namespace {

inline uint32_t f32_tile_word(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * 256 + (row & 15) * 16 + (col & 15);
}

// Byte offset of datum (row, col) in a 32x32 tile of `eb`-byte datums.
inline uint32_t tile_datum_offset(uint32_t row, uint32_t col, uint32_t eb) {
    return ((((row >> 4) * 2 + (col >> 4)) * 256) + (row & 15) * 16 + (col & 15)) * eb;
}

template <uint32_t eb>
inline void copy_datums(uint32_t dst, uint32_t src, uint32_t n) {
    if constexpr (eb == 4) {
        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
        for (uint32_t i = 0; i < n; ++i) {
            d[i] = s[i];
        }
    } else {
        volatile tt_l1_ptr uint16_t* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst);
        volatile tt_l1_ptr uint16_t* s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src);
        for (uint32_t i = 0; i < n; ++i) {
            d[i] = s[i];
        }
    }
}

}  // namespace

void kernel_main() {
    const WorkerId w = my_worker_id();
    if (!w.active) {
        return;
    }
    constexpr auto x_args = TensorAccessorArgs<WORKER_CT_ARGS>();
    constexpr auto dst_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto out_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();

    const uint32_t p = get_common_arg_val<uint32_t>(CRT_P);
    const uint32_t table = get_common_arg_val<uint32_t>(CRT_TABLE_ADDR);
    const uint32_t fwd_hub = get_common_arg_val<uint32_t>(CRT_FWD_HUB);
    const uint32_t bwd_hub = get_common_arg_val<uint32_t>(CRT_BWD_HUB);
    constexpr uint32_t slot_bytes = R * K * nstats * RECORD_BYTES;
    constexpr uint32_t f32_tile_bytes = TILE * TILE * 4;

    // ---- publish_records_block ----
    {
        const uint32_t n = w.crt * nstats;
        {
            AGRN_ZONE("wr_wait_partial");
            cb_wait_front(cb_partial, n);
        }
        AGRN_ZONE("wr_records");
        const uint32_t src_base = get_read_ptr(cb_partial);
        const uint32_t staging = get_write_ptr(cb_record_staging);
        for (uint32_t s = 0; s < nstats; ++s) {
            for (uint32_t r = 0; r < w.crt; ++r) {
                const uint32_t t = s * w.crt + r;  // cb_partial is pushed stat-major: [stat][row]
                asm volatile("" ::: "memory");
                const uint32_t* tile = reinterpret_cast<const uint32_t*>(src_base + t * f32_tile_bytes);
                const uint32_t rec_addr = staging + t * RECORD_BYTES;
                uint32_t* rec = reinterpret_cast<uint32_t*>(rec_addr);
#pragma GCC unroll 16
                for (uint32_t i = 0; i < FACE; ++i) {  // column 0: face 0 rows 0..15, face 2 rows 16..31
                    rec[i] = tile[i * FACE];
                    rec[FACE + i] = tile[512 + i * FACE];
                }
                asm volatile("" ::: "memory");
                // Table layout per (slot, tile-row, stat): [values 0..15 of members 0..K-1][values 16..31 of members
                // 0..K-1], 64 B each — a gathering worker lands a slot's members as consecutive tile rows with one
                // contiguous read per half (see the reader).
                const uint32_t blk = table + p * slot_bytes + ((w.row_start + r) * nstats + s) * K * RECORD_BYTES;
                for (uint32_t h = 0; h < 2; ++h) {
                    const uint32_t off = blk + (h * K + w.m) * (RECORD_BYTES / 2);
                    const uint32_t src = rec_addr + h * (RECORD_BYTES / 2);
                    noc_async_write(src, get_noc_addr(fwd_hub & 0xFFFF, fwd_hub >> 16, off), RECORD_BYTES / 2);
                    noc_async_write(src, get_noc_addr(bwd_hub & 0xFFFF, bwd_hub >> 16, off), RECORD_BYTES / 2);
                }
            }
        }
        noc_async_write_barrier();
        const uint32_t sem = get_semaphore(sem_records_local);
        noc_semaphore_inc(get_noc_addr(fwd_hub & 0xFFFF, fwd_hub >> 16, sem), 1);
        noc_semaphore_inc(get_noc_addr(bwd_hub & 0xFFFF, bwd_hub >> 16, sem), 1);
        cb_pop_front(cb_partial, n);
    }

    // ---- store_own_block ----
    {
        {
            AGRN_ZONE("wr_wait_out");
            cb_wait_front(cb_out_block, 1);
        }
        AGRN_ZONE("wr_own");
        const auto dst = TensorAccessor(dst_args, get_common_arg_val<uint32_t>(CRT_DST_ADDR), out_page_bytes);
        for (uint32_t r = 0; r < w.crt; ++r) {
            const uint32_t tr = w.row_start + r;
            const uint32_t first = is_ru ? (p * R + tr) * Wlt + w.w_start : tr * Wt + p * Wlt + w.w_start;
            for (uint32_t j = 0; j < w.cw; ++j) {
                cb_wait_front(cb_out_block, 1);
                noc_async_write(get_read_ptr(cb_out_block), dst.get_noc_addr(first + j), out_page_bytes);
                // popped pages stay in place: cb_out_block holds the whole block and nothing pushes again
                cb_pop_front(cb_out_block, 1);
            }
        }
        noc_async_write_barrier();
        const uint32_t sem = get_semaphore(sem_own_ready);
        for (uint32_t i = 0; i < n_ports; ++i) {
            const uint32_t c = get_common_arg_val<uint32_t>(CRT_PORT_NOC + i);
            noc_semaphore_inc(get_noc_addr(c & 0xFFFF, c >> 16, sem), 1);
        }
        noc_async_atomic_barrier();
    }

    // ---- assemble_block (R_U) ----
    if constexpr (is_ru) {
        volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_landed));
        {
            AGRN_ZONE("wr_wait_landed");
            noc_semaphore_wait_min(landed, n_ports);
        }
        AGRN_ZONE("wr_assemble");
        const auto staging = TensorAccessor(dst_args, get_common_arg_val<uint32_t>(CRT_DST_ADDR), out_page_bytes);
        const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(CRT_OUT_ADDR), out_page_bytes);
        constexpr uint32_t W = G * W_local;
        constexpr uint32_t eb = out_elem_bytes;
        constexpr uint32_t cwo = (Wt + K - 1) / K;  // output tile-columns per member
        const uint32_t oc0 = w.m * cwo;
        const uint32_t oc1 = (oc0 + cwo) < Wt ? oc0 + cwo : Wt;
        const uint32_t scratch = get_write_ptr(cb_assemble);
        const uint32_t out_tile = scratch + assemble_src_max * out_page_bytes;
        for (uint32_t r = 0; r < w.crt; ++r) {
            const uint32_t tr = w.row_start + r;
            for (uint32_t oc = oc0; oc < oc1; ++oc) {
                // source tiles of this output tile, in column order (consecutive columns share a source)
                uint32_t src_page[assemble_src_max];
                uint32_t n_src = 0;
                for (uint32_t c = 0; c < TILE; ++c) {
                    const uint32_t g = oc * TILE + c;
                    if (g >= W) {
                        break;
                    }
                    const uint32_t q = g / W_local;
                    const uint32_t lc = g - q * W_local;
                    const uint32_t page = (q * R + tr) * Wlt + lc / TILE;
                    if (n_src == 0 || src_page[n_src - 1] != page) {
                        src_page[n_src] = page;
                        noc_async_read(staging.get_noc_addr(page), scratch + n_src * out_page_bytes, out_page_bytes);
                        ++n_src;
                    }
                }
                noc_async_read_barrier();
                // element-shifted copy: segments [c, c + n) with one source tile, contiguous source columns, and
                // no 16-column face boundary crossed on either side
                uint32_t si = 0;
                uint32_t c = 0;
                while (c < TILE) {
                    const uint32_t g = oc * TILE + c;
                    if (g >= W) {
                        // columns past W: zero (padding of the last output tile)
                        for (uint32_t row = 0; row < TILE; ++row) {
                            for (uint32_t cc = c; cc < TILE; ++cc) {
                                const uint32_t d = out_tile + tile_datum_offset(row, cc, eb);
                                if constexpr (eb == 4) {
                                    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d) = 0;
                                } else {
                                    *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(d) = 0;
                                }
                            }
                        }
                        break;
                    }
                    const uint32_t q = g / W_local;
                    const uint32_t lc = g - q * W_local;
                    const uint32_t page = (q * R + tr) * Wlt + lc / TILE;
                    while (src_page[si] != page) {
                        ++si;
                    }
                    const uint32_t sc = lc % TILE;
                    uint32_t n = FACE - (c % FACE);
                    const uint32_t n_src_face = FACE - (sc % FACE);
                    n = n < n_src_face ? n : n_src_face;
                    const uint32_t n_slice = (q + 1) * W_local - g;  // columns left in this slot
                    n = n < n_slice ? n : n_slice;
                    const uint32_t n_w = W - g;
                    n = n < n_w ? n : n_w;
                    const uint32_t src_tile = scratch + si * out_page_bytes;
                    for (uint32_t row = 0; row < TILE; ++row) {
                        copy_datums<eb>(
                            out_tile + tile_datum_offset(row, c, eb), src_tile + tile_datum_offset(row, sc, eb), n);
                    }
                    c += n;
                }
                noc_async_write(out_tile, out.get_noc_addr(tr * Wt + oc), out_page_bytes);
                noc_async_write_barrier();
            }
        }
    }
}
