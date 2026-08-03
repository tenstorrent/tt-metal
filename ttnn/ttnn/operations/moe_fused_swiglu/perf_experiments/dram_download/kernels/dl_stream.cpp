// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DRAM DOWNLOAD FLOOR — ONE symmetric streamer, instantiated on BOTH data-movement RISCs.
//
// The first version of this bench hard-split the work by matrix (reader took W_gate, writer took
// W_up + W_down), which is a 32/68 BYTE split — the writer carried 2.14x the reader's bytes, so the
// measurement was reporting a lopsided schedule rather than the achievable rate. This kernel takes a
// per-matrix ROW RANGE instead, so the host can put any fraction on either RISC: all on NOC_0, all on
// NOC_1, or a balanced split derived from each NoC's measured solo rate.
//
// `DM_DEDICATED_NOC` binds the reader to NOC_0 and the writer to NOC_1, so which KERNEL issues a read
// IS the NoC choice. Row ranges are disjoint, so both RISCs write to non-overlapping L1 and the
// landing addresses are derived from the absolute row index — no coordination needed.
//
// All reads go in flight and ONE barrier closes them: the most favourable schedule, hence a ceiling
// on rate. Reads are issued matrix-INTERLEAVED (one row of each per pass) rather than matrix-by-matrix
// so a single RISC's stream is spread across the three buffers instead of hammering one at a time.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t HID_T = get_compile_time_arg_val(0);       // hidden tiles (gate/up row pitch)
constexpr uint32_t EMB_T = get_compile_time_arg_val(1);       // emb tiles (W_down row pitch)
constexpr uint32_t BFP4_TILE = get_compile_time_arg_val(2);   // 576
constexpr uint32_t X_ELEM = get_compile_time_arg_val(3);      // 2 = bf16 row-major, 0 = bfp8 tiles
constexpr uint32_t X_PAGE = get_compile_time_arg_val(4);      // accessor page: emb*2 (RM) or 1088 (tile)
constexpr uint32_t INTERLEAVE = get_compile_time_arg_val(5);  // 1 = round-robin the three matrices
constexpr uint32_t TILE_H = 32;

// PRIVATE CB SET PER RISC — load-bearing, not tidiness. A circular buffer has ONE write pointer and
// exactly one producer by contract; having both data-movement RISCs call `cb_reserve_back` on the same
// CB hung the device (and would have had them writing the same L1 addresses even if it had not). The
// host hands the reader {0,1,2,3} and the writer {4,5,6,7}.
constexpr uint32_t CB_WG = get_compile_time_arg_val(6);
constexpr uint32_t CB_WU = get_compile_time_arg_val(7);
constexpr uint32_t CB_WD = get_compile_time_arg_val(8);
constexpr uint32_t CB_X = get_compile_time_arg_val(9);

constexpr uint32_t TA_BASE = 10;
constexpr auto wg_args = TensorAccessorArgs<TA_BASE>();
constexpr auto wu_args = TensorAccessorArgs<wg_args.next_compile_time_args_offset()>();
constexpr auto wd_args = TensorAccessorArgs<wu_args.next_compile_time_args_offset()>();
constexpr auto x_args = TensorAccessorArgs<wd_args.next_compile_time_args_offset()>();

void kernel_main() {
    uint32_t i = 0;
    const uint32_t wg_addr = get_arg_val<uint32_t>(i++);
    const uint32_t wu_addr = get_arg_val<uint32_t>(i++);
    const uint32_t wd_addr = get_arg_val<uint32_t>(i++);
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t kstart = get_arg_val<uint32_t>(i++);   // my grid row's first emb K-tile
    const uint32_t hstart = get_arg_val<uint32_t>(i++);   // my grid column's first hidden tile
    const uint32_t hn = get_arg_val<uint32_t>(i++);       // my hidden-tile count
    const uint32_t ecstart = get_arg_val<uint32_t>(i++);  // my first emb OUTPUT tile
    const uint32_t ec = get_arg_val<uint32_t>(i++);       // my emb-output tile count
    const uint32_t wg_r0 = get_arg_val<uint32_t>(i++);    // MY row range of W_gate
    const uint32_t wg_rn = get_arg_val<uint32_t>(i++);
    const uint32_t wu_r0 = get_arg_val<uint32_t>(i++);
    const uint32_t wu_rn = get_arg_val<uint32_t>(i++);
    const uint32_t wd_r0 = get_arg_val<uint32_t>(i++);
    const uint32_t wd_rn = get_arg_val<uint32_t>(i++);
    const uint32_t kr = get_arg_val<uint32_t>(i++);  // my TOTAL K-tile count (for x slicing)
    const uint32_t x_rows = get_arg_val<uint32_t>(i++);
    const uint32_t x_row0 = get_arg_val<uint32_t>(i++);

    const auto wg_acc = TensorAccessor(wg_args, wg_addr, BFP4_TILE);
    const auto wu_acc = TensorAccessor(wu_args, wu_addr, BFP4_TILE);
    const auto wd_acc = TensorAccessor(wd_args, wd_addr, BFP4_TILE);

    const uint32_t gu_run = hn * BFP4_TILE;
    const uint32_t wd_run = ec * BFP4_TILE;
    uint32_t wg_wp = 0, wu_wp = 0, wd_wp = 0;
    if (wg_rn) {
        cb_reserve_back(CB_WG, wg_rn * hn);
        wg_wp = get_write_ptr(CB_WG);
    }
    if (wu_rn) {
        cb_reserve_back(CB_WU, wu_rn * hn);
        wu_wp = get_write_ptr(CB_WU);
    }
    if (wd_rn) {
        cb_reserve_back(CB_WD, wd_rn * ec);
        wd_wp = get_write_ptr(CB_WD);
    }

    if constexpr (INTERLEAVE) {
        // Round-robin the three streams so this RISC's outstanding requests are spread over three
        // buffers (and therefore three bank phases) instead of walking one buffer to completion.
        const uint32_t rounds = (wg_rn > wu_rn ? wg_rn : wu_rn) > wd_rn ? (wg_rn > wu_rn ? wg_rn : wu_rn) : wd_rn;
        for (uint32_t r = 0; r < rounds; ++r) {
            if (r < wg_rn) {
                noc_async_read(wg_acc.get_noc_addr((kstart + wg_r0 + r) * HID_T + hstart), wg_wp + r * gu_run, gu_run);
            }
            if (r < wu_rn) {
                noc_async_read(wu_acc.get_noc_addr((kstart + wu_r0 + r) * HID_T + hstart), wu_wp + r * gu_run, gu_run);
            }
            if (r < wd_rn) {
                noc_async_read(wd_acc.get_noc_addr((wd_r0 + r) * EMB_T + ecstart), wd_wp + r * wd_run, wd_run);
            }
        }
    } else {
        for (uint32_t r = 0; r < wg_rn; ++r) {
            noc_async_read(wg_acc.get_noc_addr((kstart + wg_r0 + r) * HID_T + hstart), wg_wp + r * gu_run, gu_run);
        }
        for (uint32_t r = 0; r < wu_rn; ++r) {
            noc_async_read(wu_acc.get_noc_addr((kstart + wu_r0 + r) * HID_T + hstart), wu_wp + r * gu_run, gu_run);
        }
        for (uint32_t r = 0; r < wd_rn; ++r) {
            noc_async_read(wd_acc.get_noc_addr((wd_r0 + r) * EMB_T + ecstart), wd_wp + r * wd_run, wd_run);
        }
    }

    if (x_rows) {
        const auto x_acc = TensorAccessor(x_args, x_addr, X_PAGE);
        cb_reserve_back(CB_X, (X_ELEM == 2) ? (x_rows * TILE_H) : (x_rows * kr));
        const uint32_t xp = get_write_ptr(CB_X);
        if constexpr (X_ELEM == 2) {
            const uint32_t slice = kr * TILE_H * 2;
            for (uint32_t r = 0; r < x_rows; ++r) {
                for (uint32_t s = 0; s < TILE_H; ++s) {
                    noc_async_read(
                        x_acc.get_noc_addr((x_row0 + r) * TILE_H + s, kstart * TILE_H * 2),
                        xp + (r * TILE_H + s) * slice,
                        slice);
                }
            }
        } else {
            for (uint32_t r = 0; r < x_rows; ++r) {
                for (uint32_t t = 0; t < kr; ++t) {
                    noc_async_read(
                        x_acc.get_noc_addr((x_row0 + r) * EMB_T + kstart + t), xp + (r * kr + t) * X_PAGE, X_PAGE);
                }
            }
        }
    }

    noc_async_read_barrier();
}
