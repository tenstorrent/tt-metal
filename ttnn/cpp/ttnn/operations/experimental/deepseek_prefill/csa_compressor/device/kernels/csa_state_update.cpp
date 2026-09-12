// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"

namespace {

constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kTileElems = 1024;

// Scratch slots. The output tile pair being patched, then for the half currently being scattered the
// one or two source tile rows its tokens span, and its bias tile. The kv and gate pairs are indexed
// as kScratchKv + which and kScratchGate + which, where which selects the low or high source row.
constexpr uint32_t kScratchOutKv = 0;
constexpr uint32_t kScratchOutScore = 1;
constexpr uint32_t kScratchKv = 2;
constexpr uint32_t kScratchGate = 4;
constexpr uint32_t kScratchBias = 6;
constexpr uint32_t kScratchTiles = 7;

inline uint32_t tile_offset(uint32_t row, uint32_t col) {
    const uint32_t face = (row / 16) * 2 + col / 16;
    return face * 256 + (row % 16) * 16 + col % 16;
}

inline float bf16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline uint16_t float_to_bf16_rne(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

// One tile row is two runs of 16 contiguous elements, one per face column, and both start on a
// 32-byte boundary. Moving them as words keeps the copy to 16 stores per row.
inline void copy_tile_row(
    volatile tt_l1_ptr uint16_t* destination_tile,
    volatile tt_l1_ptr uint16_t* source_tile,
    uint32_t destination_row,
    uint32_t source_row) {
    for (uint32_t face_col = 0; face_col < 32; face_col += 16) {
        volatile tt_l1_ptr uint32_t* destination =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination_tile + tile_offset(destination_row, face_col));
        volatile tt_l1_ptr uint32_t* source =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(source_tile + tile_offset(source_row, face_col));
        for (uint32_t word = 0; word < 8; ++word) {
            destination[word] = source[word];
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t kv_addr = get_arg_val<uint32_t>(0);
    const uint32_t gate_addr = get_arg_val<uint32_t>(1);
    const uint32_t bias_addr = get_arg_val<uint32_t>(2);
    const uint32_t base_kv_addr = get_arg_val<uint32_t>(3);
    const uint32_t base_score_addr = get_arg_val<uint32_t>(4);
    const uint32_t output_kv_addr = get_arg_val<uint32_t>(5);
    const uint32_t output_score_addr = get_arg_val<uint32_t>(6);
    const uint32_t local_valid = get_arg_val<uint32_t>(7);
    const uint32_t absolute_start = get_arg_val<uint32_t>(8);
    const uint32_t state_tiles = get_arg_val<uint32_t>(9);
    const uint32_t first_state_tile = get_arg_val<uint32_t>(10);

    constexpr uint32_t input_width_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t state_width_tiles = get_compile_time_arg_val(1);
    constexpr auto kv_args = TensorAccessorArgs<2>();
    constexpr auto gate_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();
    constexpr auto base_kv_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr auto base_score_args = TensorAccessorArgs<base_kv_args.next_compile_time_args_offset()>();
    constexpr auto output_kv_args = TensorAccessorArgs<base_score_args.next_compile_time_args_offset()>();
    constexpr auto output_score_args = TensorAccessorArgs<output_kv_args.next_compile_time_args_offset()>();

    const auto kv = TensorAccessor(kv_args, kv_addr);
    const auto gate = TensorAccessor(gate_args, gate_addr);
    const auto bias = TensorAccessor(bias_args, bias_addr);
    const auto base_kv = TensorAccessor(base_kv_args, base_kv_addr);
    const auto base_score = TensorAccessor(base_score_args, base_score_addr);
    const auto output_kv = TensorAccessor(output_kv_args, output_kv_addr);
    const auto output_score = TensorAccessor(output_score_args, output_score_addr);

    constexpr uint32_t scratch_cb = tt::CBIndex::c_3;
    CircularBuffer scratch(scratch_cb);
    scratch.reserve_back(kScratchTiles);
    volatile tt_l1_ptr uint16_t* memory = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch.get_write_ptr());
    Noc noc;

    // Rows 0-7 and 32-39 of the 64-row slab are live; every other row passes through from the base
    // state. A live row wants the newest token whose position satisfies
    // `position % 4 == slot && (position / 4) & 1 == parity`, which is the single condition
    // `position % 8 == 4 * parity + slot`. The sixteen live rows cover all eight residues twice, so
    // they draw from the last eight tokens of the chunk -- and eight consecutive tokens straddle at
    // most two source tile rows. That is what lets one pair of reads per half feed all four of its
    // rows, instead of the three reads and a full round trip each row used to cost.
    const uint32_t last_token = local_valid > 0 ? local_valid - 1 : 0;
    const uint32_t first_token = local_valid >= 8 ? local_valid - 8 : 0;
    const uint32_t source_row_lo = first_token / 32;
    const uint32_t source_row_hi = last_token / 32;

    for (uint32_t tile = first_state_tile; tile < first_state_tile + state_tiles; ++tile) {
        noc.async_read(base_kv, scratch, kTileBytes, {.page_id = tile}, {.offset_bytes = kScratchOutKv * kTileBytes});
        noc.async_read(
            base_score, scratch, kTileBytes, {.page_id = tile}, {.offset_bytes = kScratchOutScore * kTileBytes});
        noc.async_read_barrier();
        invalidate_l1_cache();

        const uint32_t tile_row = tile / state_width_tiles;
        const uint32_t feature_tile = tile % state_width_tiles;

        if (local_valid > 0) {
            for (uint32_t half = 0; half < 2; ++half) {
                // Ca occupies rows 0-3 of the tile and reads the left half of the projection, Cb rows
                // 4-7 and the right half. A window writes its Ca into the slab of the opposite parity,
                // which is why the two halves of one tile row look for opposite parities.
                const uint32_t parity = half == 0 ? 1 - tile_row : tile_row;
                const uint32_t column_tile = half * state_width_tiles + feature_tile;

                noc.async_read(
                    kv,
                    scratch,
                    kTileBytes,
                    {.page_id = source_row_lo * input_width_tiles + column_tile},
                    {.offset_bytes = kScratchKv * kTileBytes});
                noc.async_read(
                    gate,
                    scratch,
                    kTileBytes,
                    {.page_id = source_row_lo * input_width_tiles + column_tile},
                    {.offset_bytes = kScratchGate * kTileBytes});
                if (source_row_hi != source_row_lo) {
                    noc.async_read(
                        kv,
                        scratch,
                        kTileBytes,
                        {.page_id = source_row_hi * input_width_tiles + column_tile},
                        {.offset_bytes = (kScratchKv + 1) * kTileBytes});
                    noc.async_read(
                        gate,
                        scratch,
                        kTileBytes,
                        {.page_id = source_row_hi * input_width_tiles + column_tile},
                        {.offset_bytes = (kScratchGate + 1) * kTileBytes});
                }
                noc.async_read(
                    bias, scratch, kTileBytes, {.page_id = column_tile}, {.offset_bytes = kScratchBias * kTileBytes});
                noc.async_read_barrier();
                invalidate_l1_cache();

                for (uint32_t slot = 0; slot < 4; ++slot) {
                    // When local_valid is 0 this loop is skipped entirely; otherwise `back` is in
                    // [0, 8) and only overshoots the front of the chunk when no token qualifies.
                    const uint32_t target = 4 * parity + slot;
                    const uint32_t back = (absolute_start + last_token + 8 - target) % 8;
                    if (back >= local_valid) {
                        continue;
                    }
                    const uint32_t source_token = last_token - back;
                    const uint32_t which = source_token / 32 == source_row_lo ? 0 : 1;
                    const uint32_t source_row = source_token % 32;
                    const uint32_t row_in_tile = 4 * half + slot;

                    copy_tile_row(
                        memory + kScratchOutKv * kTileElems,
                        memory + (kScratchKv + which) * kTileElems,
                        row_in_tile,
                        source_row);

                    // The score half stays element-wise: the bias add has to round in software bf16
                    // to stay bit-exact with the reference state.
                    for (uint32_t col = 0; col < 32; ++col) {
                        const uint32_t src = tile_offset(source_row, col);
                        const uint32_t bias_src = tile_offset(slot, col);
                        const float score = bf16_to_float(memory[(kScratchGate + which) * kTileElems + src]) +
                                            bf16_to_float(memory[kScratchBias * kTileElems + bias_src]);
                        memory[kScratchOutScore * kTileElems + tile_offset(row_in_tile, col)] =
                            float_to_bf16_rne(score);
                    }
                }
            }
        }

        noc.async_write(
            use<CircularBuffer::AddrSelector::WRITE_PTR>(scratch),
            output_kv,
            kTileBytes,
            {.offset_bytes = kScratchOutKv * kTileBytes},
            {.page_id = tile});
        noc.async_write(
            use<CircularBuffer::AddrSelector::WRITE_PTR>(scratch),
            output_score,
            kTileBytes,
            {.offset_bytes = kScratchOutScore * kTileBytes},
            {.page_id = tile});
        noc.async_write_barrier();
    }
    scratch.push_back(kScratchTiles);
}
