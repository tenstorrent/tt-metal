// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The accumulator reader's side of a quantized row (common/kernels/quantized_row.hpp): a row landed in this core's L1
// (a landing slot, or the L1 hand-off) into the CB pages the accumulator compute reads (quantized_row_weigh.hpp).
//
//   Bf16     the row CB: the values (a 2048-B page a flat tile) - today's copy.
//   Int8     the row CB as Bfp8_b tiles: a flat tile's 64 exponent bytes, then its 1024 values (a 1088-B page).
//   Fp8E4m3  the row CB: the codes (a 1024-B page a flat tile); the factor CB: an FP16 tile a flat tile, DEST row r's
//            factor (its exponent byte's scale x 2^kFp8FactorShift, clamped to FP16's normal range) in face row r's
//            column 0 (the other columns are never read: zeroed once).
//
// Usage (one RowStager a reader; the CBs' page sizes: row_page_bytes / factor_page_bytes):
//
//     quantized_row::RowStager<format, H, producers, cb_row, cb_factor> stager;   // cb_factor: Fp8E4m3 only
//     stager.init();                    // once
//     ...
//     stager.stage(slot);               // a landed row at L1 address `slot`: reserves, starts the reads, writes the
//                                       // exponents / factors
//     noc_async_read_barrier();         // (with whatever else the reader reads)
//     stager.push();                    // the row (and factor) pages to the compute
//
// The slot holds the row as it landed: row_bytes(format, H, producers) bytes, the trailer after them.
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/quantized_row.hpp"

namespace quantized_row {

template <Format format, uint32_t H, uint32_t producers, uint32_t cb_row, uint32_t cb_factor = 0>
struct RowStager {
    static_assert(valid_shape(format, H, producers), "quantized_row: a row shape the format cannot carry");
    static constexpr uint32_t kTiles = flat_tiles(H);
    static constexpr uint32_t kData = data_bytes(format, H);
    static constexpr uint32_t kCellWords = cell_exponents(H, producers) / 4;  // a cell's exponent words

    // Fp8E4m3: an exponent byte -> its FP16 factor's bits
    uint16_t factor_bits[format == Format::Fp8E4m3 ? 256 : 1];

    // Once: Fp8E4m3's factor table, and its factor CB zeroed (the columns the compute never reads stay finite)
    void init() {
        if constexpr (format == Format::Fp8E4m3) {
            for (int32_t byte = 0; byte < 256; ++byte) {
                int32_t e = byte - 127 + kFp8FactorShift + 15;  // FP16 exponent field of 2^(byte - 127 + shift)
                e = e < 1 ? 1 : (e > 30 ? 30 : e);
                factor_bits[byte] = static_cast<uint16_t>(e << 10);
            }
            const uint32_t pages = get_local_cb_interface(cb_factor).fifo_num_pages;
            cb_reserve_back(cb_factor, pages);
            volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_factor));
            for (uint32_t w = 0; w < pages * factor_page_bytes(format) / 4; ++w) {
                f[w] = 0;
            }
        }
    }

    // One landed row: reserve its pages, start its value reads (NoC, local), write its exponents / factors
    void stage(uint32_t slot) {
        cb_reserve_back(cb_row, kTiles);
        const uint32_t row = get_write_ptr(cb_row);
        if constexpr (format == Format::Int8) {
            // a Bfp8_b tile a flat tile: the values behind its 64-byte exponent section
            for (uint32_t p = 0; p < kTiles; ++p) {
                noc_async_read(
                    get_noc_addr(slot + p * kFlatTileValues),
                    row + p * row_page_bytes(format) + kTileGroups,
                    kFlatTileValues);
            }
            // exponent word w (bytes 4w .. 4w + 3) = cell w / kCellWords, word w % kCellWords: flat tile w / 16's
            // exponent section, word w % 16
            volatile tt_l1_ptr uint32_t* cell = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + kData);
            volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row);
            uint32_t w = 0;
            for (uint32_t c = 0; c < producers; ++c, cell += kCellBytes / 4) {
#pragma GCC unroll 4
                for (uint32_t j = 0; j < kCellWords; ++j, ++w) {
                    page[(w / (kTileGroups / 4)) * (row_page_bytes(format) / 4) + w % (kTileGroups / 4)] = cell[j];
                }
            }
        } else {
            noc_async_read(get_noc_addr(slot), row, kData);
        }
        if constexpr (format == Format::Fp8E4m3) {
            // exponent byte k (values 16k .. 16k + 15): flat tile k / 64, DEST row k % 64, column 0 - halfword 16k of
            // the row's factor pages (a page: 64 face rows of 32 bytes)
            cb_reserve_back(cb_factor, kTiles);
            volatile tt_l1_ptr uint32_t* cell = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + kData);
            volatile tt_l1_ptr uint16_t* f = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_factor));
            for (uint32_t c = 0; c < producers; ++c, cell += kCellBytes / 4) {
#pragma GCC unroll 4
                for (uint32_t j = 0; j < kCellWords; ++j, f += 64) {
                    const uint32_t word = cell[j];
                    f[0] = factor_bits[word & 0xFF];
                    f[16] = factor_bits[(word >> 8) & 0xFF];
                    f[32] = factor_bits[(word >> 16) & 0xFF];
                    f[48] = factor_bits[word >> 24];
                }
            }
        }
    }

    // After the reads landed: the row (and its factors) to the compute
    void push() {
        if constexpr (format == Format::Fp8E4m3) {
            cb_push_back(cb_factor, kTiles);
        }
        cb_push_back(cb_row, kTiles);
    }
};

}  // namespace quantized_row
