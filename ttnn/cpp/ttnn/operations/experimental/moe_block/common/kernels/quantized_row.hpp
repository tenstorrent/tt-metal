// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A routed expert's output row on the combine's wire: its values, then (quantized) its scales, then the trailer
// (routed_experts_combine combine_formats.hpp). Constexpr only: the host, the producers (the flat expert's y writers)
// and the consumers (the combine's accumulators: routed_experts_combine kernels quantized_row_stage.hpp /
// quantized_row_weigh.hpp) share it.
//
//   Bf16     [H bf16]                                                              2H bytes      (today)
//   Fp8E4m3  [H e4m3 codes][cells]                                                  H + 16 P
//   Int8     [H sign-magnitude int8 values][cells]                                  H + 16 P
//
// P producer cores (the flat expert's down cores) each own H / P consecutive columns and write one 16-byte cell: the
// exponent bytes of its columns, one per kGroupValues (16) consecutive values, in column order (then zero padding).
// So the row carries one exponent byte per 16 values (H / 16 in all), whatever the producer's scale block: a block of
// 32 or 96 values repeats its byte 2 or 6 times. Cells are 16 bytes for the NoC (a core's write must be 16-B aligned).
//
// What a value is (s a power of two, the 16 values' scale):
//   Fp8E4m3  e4m3(code) x s, the byte = log2(s) + 127 (e8m0). The producer flushes |code| < 2^-6 to 0 (the
//            unpacker flushes e4m3 subnormals to 0 anyway).
//   Int8     (sign ? -1 : 1) x magnitude x s, magnitude 0..127, the byte = log2(s) + 133 - so the 16 values with their
//            byte are exactly a Bfp8_b face row (mantissa byte = sign | magnitude, value = magnitude / 64 x
//            2^(byte - 127)): the consumer unpacks them as Bfp8_b. The producer never writes 0x80 (-0: Bfp8_b reads a
//            zero magnitude with the sign set as -inf).
//
// GLM-5.3 (H 6144, P 32): Bf16 12288 B, Fp8E4m3 / Int8 6144 + 512 B (+ the 32-byte trailer).
//
// The consumer sees a row as flat "tiles" of 1024 consecutive values (everything it does is elementwise, so the order
// inside a tile does not matter as long as it comes back out the same): 16 consecutive values are one face row of a
// flat tile, one DEST row - and one exponent byte.
#pragma once

#include <cstdint>

namespace quantized_row {

enum class Format : uint32_t {
    Bf16 = 0,
    Fp8E4m3 = 1,
    Int8 = 2,
};

inline constexpr uint32_t kFlatTileValues = 1024;  // a flat tile: 1024 consecutive values of a row
inline constexpr uint32_t kGroupValues = 16;       // values an exponent byte covers (a face row of a flat tile)
inline constexpr uint32_t kTileGroups = kFlatTileValues / kGroupValues;  // exponent bytes a flat tile (64)
inline constexpr uint32_t kCellBytes = 16;                               // a producer core's cell

constexpr bool is_quantized(Format f) { return f != Format::Bf16; }
constexpr uint32_t value_bytes(Format f) { return f == Format::Bf16 ? 2 : 1; }

constexpr uint32_t flat_tiles(uint32_t h) { return h / kFlatTileValues; }
constexpr uint32_t data_bytes(Format f, uint32_t h) { return h * value_bytes(f); }
// a producer core's exponent bytes (its H / P columns / 16)
constexpr uint32_t cell_exponents(uint32_t h, uint32_t producers) { return h / producers / kGroupValues; }
constexpr uint32_t cells_bytes(Format f, uint32_t producers) { return is_quantized(f) ? producers * kCellBytes : 0; }
// the row's bytes before its trailer
constexpr uint32_t row_bytes(Format f, uint32_t h, uint32_t producers) {
    return data_bytes(f, h) + cells_bytes(f, producers);
}
// the row's shape is one the format can carry
constexpr bool valid_shape(Format f, uint32_t h, uint32_t producers) {
    return h % kFlatTileValues == 0 &&
           (!is_quantized(f) || (producers > 0 && h % (producers * kGroupValues) == 0 &&
                                 cell_exponents(h, producers) <= kCellBytes && cell_exponents(h, producers) % 4 == 0));
}

// ---- the consumer's CBs (quantized_row_stage.hpp fills them, quantized_row_weigh.hpp reads them) ----
// the row CB: a page a flat tile - Bf16 tiles, e4m3 tiles, or Bfp8_b tiles (64 exponent bytes, then 1024 mantissas)
constexpr uint32_t row_page_bytes(Format f) {
    return f == Format::Bf16 ? 2048 : (f == Format::Fp8E4m3 ? 1024 : kTileGroups + kFlatTileValues);
}
// Fp8E4m3's factor CB: an FP16 tile a flat tile (DEST row r's factor in face row r's column 0); none otherwise
constexpr uint32_t factor_page_bytes(Format f) { return f == Format::Fp8E4m3 ? 2048 : 0; }
// Fp8E4m3: the factor is the scale x 2^kFp8FactorShift (FP16 has 5 exponent bits: the shift keeps every scale that can
// matter in its normal range; the weigh step multiplies the 2^-shift back in). Exact for combine_snap's q = 2^-14: a
// factor clamped up from below FP16's normal range belongs to values under 2^-15.2 (their terms round to 0 either
// way), and one clamped down needs |y| > 14336.
inline constexpr int32_t kFp8FactorShift = 10;

}  // namespace quantized_row
