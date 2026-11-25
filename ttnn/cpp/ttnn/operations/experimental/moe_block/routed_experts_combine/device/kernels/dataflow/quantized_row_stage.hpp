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
// Two phases, for rows landed in this core's L1 or on another core (a flat-expert pool slot), several in flight:
//
//     quantized_row::RowStager<format, H, producers, cb_row, cb_factor> stager;   // cb_factor: Fp8E4m3 only
//     stager.init();                                   // once
//     stager.issue(slot_noc, row_l1, cells_l1);        // the reads: values -> the row's pages at row_l1 (reserved),
//                                                      // cells (kCellsBytes) -> cells_l1 (scratch)
//     noc_async_read_barrier();
//     stager.finish(row_l1, cells_l1, factor_l1);      // Int8: exponents into the row's pages; Fp8E4m3: the factor
//                                                      // pages at factor_l1 (reserved); then push the pages
//
// or, for one row in this core's L1 with the CBs' own pointers: stage(slot) (reserves, issues, waits, finishes) and
// push(). The slot holds the row as it landed: row_bytes(format, H, producers) bytes, the trailer after them.
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

    static constexpr uint32_t kCellsBytes = cells_bytes(format, producers);
    static constexpr uint32_t kFactorTileHalfwords = factor_page_bytes(format) / 2;

    // A row's reads: its values into its row pages at row_l1 (Int8: each flat tile's 1024 values behind its page's
    // 64-byte exponent section), its cells into cells_l1 (quantized)
    static void issue(uint64_t slot_noc, uint32_t row_l1, [[maybe_unused]] uint32_t cells_l1) {
        if constexpr (format == Format::Int8) {
            for (uint32_t p = 0; p < kTiles; ++p) {
                noc_async_read(
                    slot_noc + p * kFlatTileValues, row_l1 + p * row_page_bytes(format) + kTileGroups, kFlatTileValues);
            }
        } else {
            noc_async_read(slot_noc, row_l1, kData);
        }
        if constexpr (is_quantized(format)) {
            noc_async_read(slot_noc + kData, cells_l1, kCellsBytes);
        }
    }

    // After the row's reads landed: Int8 its exponents into its pages' exponent sections; Fp8E4m3 its factor pages
    // at factor_l1 (DEST row r's factor in face row r's column 0)
    void finish(
        [[maybe_unused]] uint32_t row_l1, [[maybe_unused]] uint32_t cells_l1, [[maybe_unused]] uint32_t factor_l1) {
        if constexpr (format == Format::Int8) {
            // exponent word w (bytes 4w .. 4w + 3) = cell w / kCellWords, word w % kCellWords: flat tile w / 16's
            // exponent section, word w % 16
            volatile tt_l1_ptr uint32_t* cell = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cells_l1);
            volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_l1);
            uint32_t w = 0;
            for (uint32_t c = 0; c < producers; ++c, cell += kCellBytes / 4) {
#pragma GCC unroll 4
                for (uint32_t j = 0; j < kCellWords; ++j, ++w) {
                    page[(w / (kTileGroups / 4)) * (row_page_bytes(format) / 4) + w % (kTileGroups / 4)] = cell[j];
                }
            }
        } else if constexpr (format == Format::Fp8E4m3) {
            // exponent byte k (values 16k .. 16k + 15): flat tile k / 64, DEST row k % 64, column 0 - halfword 16k
            // of the row's factor pages (a page: 64 face rows of 32 bytes)
            volatile tt_l1_ptr uint32_t* cell = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cells_l1);
            volatile tt_l1_ptr uint16_t* f = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(factor_l1);
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

    // One row in this core's L1 at `slot`, through the CBs' own write pointers: reserve, read, finish (the cells read
    // in place: no scratch). Then push().
    void stage(uint32_t slot) {
        cb_reserve_back(cb_row, kTiles);
        const uint32_t row = get_write_ptr(cb_row);
        uint32_t factor = 0;
        if constexpr (format == Format::Fp8E4m3) {
            cb_reserve_back(cb_factor, kTiles);
            factor = get_write_ptr(cb_factor);
        }
        if constexpr (format == Format::Int8) {
            for (uint32_t p = 0; p < kTiles; ++p) {
                noc_async_read(
                    get_noc_addr(slot + p * kFlatTileValues),
                    row + p * row_page_bytes(format) + kTileGroups,
                    kFlatTileValues);
            }
        } else {
            noc_async_read(get_noc_addr(slot), row, kData);
        }
        finish(row, slot + kData, factor);
    }

    // After stage() and the reads' barrier: the row (and its factors) to the compute
    void push() {
        if constexpr (format == Format::Fp8E4m3) {
            cb_push_back(cb_factor, kTiles);
        }
        cb_push_back(cb_row, kTiles);
    }
};

}  // namespace quantized_row
