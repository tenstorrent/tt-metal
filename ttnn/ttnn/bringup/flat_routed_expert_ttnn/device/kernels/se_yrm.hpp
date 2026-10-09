// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Row-major bf16 y (SE_Y_RM, with SE_DYN): y is [rows, H] bf16 row major, one page (H * 2 bytes) per flat row. The
// down compute pack-untilizes each row tile of its [MT x PCD] block and pushes it as PCD pages (32 rows x PCD * 32
// bf16, row after row); the writer sends each token row's PCD * 64 bytes to y[row, col0 * 32 ..], skipping rows at or
// past the entry's token count. Up to ROWS row tiles (the out CB's capacity) are in flight, each on its own write
// transaction id (TRID0 + k); they retire in order once their id is flushed. A virtual expert (entry a, sub-block s)
// is complete after its row tiles that hold tokens, min(MT, ceil((cnt - s MT 32) / 32)): the count the compute uses.
// The caller resets the write transaction id to 0 before the kernel exits.
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "se_dyn.hpp"

template <uint32_t out_cb, uint32_t pcd, uint32_t mt, uint32_t ht>
struct SeYRmWriter {
    static constexpr uint32_t seg = pcd * 64;           // one token row of this core's columns (bytes)
    static constexpr uint32_t tile_bytes = pcd * 2048;  // one row tile in the out CB
    static constexpr uint32_t rows = SE_Y_RM_ROWS;      // out CB capacity in row tiles (host: yrm_rows)
    static constexpr uint32_t trid0 = 8;
    static_assert(rows >= 1 && trid0 + rows <= 16);
    InterleavedAddrGen<true> yg;
    uint32_t col_off;
    uint32_t base = 0;  // the out CB's start (its read pointer before anything is consumed)
    // issue cursor (entry, sub-block, row tile) and the in-order retire cursor
    uint32_t ia = 0, is = 0, ir = 0, issued = 0;
    uint32_t da = 0, ds = 0, dr = 0, retired = 0;

    SeYRmWriter(uint32_t y_addr, uint32_t col0) :
        yg{.bank_base_address = y_addr, .page_size = ht * 64}, col_off(col0 * 64), base(get_read_ptr(out_cb)) {}

    static uint32_t rows_v(const SeDyn& dyn, uint32_t a, uint32_t s) {
        const uint32_t left = dyn.cnt[a] - s * mt * 32;
        return left >= mt * 32 ? mt : (left + 31) / 32;
    }

    // Issue the next row tile's writes if the compute has pushed it and a transaction id is free; true if issued.
    bool issue(const SeDyn& dyn) {
        if (issued - retired == rows || !cb_pages_available_at_front(out_cb, (issued - retired + 1) * pcd)) {
            return false;
        }
        const uint32_t src = base + (issued % rows) * tile_bytes;
        const uint32_t row_base = is * mt * 32 + ir * 32;
        const uint32_t left = dyn.cnt[ia] - row_base;
        const uint32_t n = left < 32 ? left : 32;
        const uint32_t row0 = dyn.off[ia] + row_base;
        const uint32_t trid = trid0 + issued % rows;
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_write_one_packet_with_trid(src + i * seg, get_noc_addr(row0 + i, yg, col_off), seg, trid);
        }
        ++issued;
        if (++ir == rows_v(dyn, ia, is)) {
            ir = 0;
            if (++is == dyn.subs[ia]) {
                is = 0;
                ++ia;
            }
        }
        return true;
    }

    // Retire the oldest issued row tile once its writes are flushed (frees its pages); returns the number of virtual
    // experts completed by it (0 or 1).
    uint32_t retire(const SeDyn& dyn) {
        if (retired == issued ||
            !ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, trid0 + retired % rows)) {
            return 0;
        }
        cb_pop_front(out_cb, pcd);
        ++retired;
        if (++dr == rows_v(dyn, da, ds)) {
            dr = 0;
            if (++ds == dyn.subs[da]) {
                ds = 0;
                ++da;
            }
            return 1;
        }
        return 0;
    }
};
