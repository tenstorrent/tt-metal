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

// Scratch slots. One source tile of kv and gate at a time, plus the predecessor state pair that
// window 0 reads. The position bias goes straight into its own CB for the compute kernel.
constexpr uint32_t kScratchKv = 0;
constexpr uint32_t kScratchGate = 1;
constexpr uint32_t kScratchPredecessorKv = 2;
constexpr uint32_t kScratchPredecessorScore = 3;
constexpr uint32_t kScratchTiles = 4;

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
// 32-byte boundary. Moving them as words keeps the scatter to 16 stores per row.
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
    const uint32_t predecessor_kv_addr = get_arg_val<uint32_t>(3);
    const uint32_t predecessor_score_addr = get_arg_val<uint32_t>(4);
    const uint32_t output_tiles = get_arg_val<uint32_t>(5);
    const uint32_t complete_windows = get_arg_val<uint32_t>(6);
    const uint32_t absolute_start = get_arg_val<uint32_t>(7);
    const uint32_t first_output_tile = get_arg_val<uint32_t>(8);

    constexpr uint32_t candidate_kv_cb = get_compile_time_arg_val(0);
    constexpr uint32_t candidate_score_cb = get_compile_time_arg_val(1);
    constexpr uint32_t scratch_cb = get_compile_time_arg_val(2);
    constexpr uint32_t input_width_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t output_width_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t ca_bias_cb = get_compile_time_arg_val(5);
    constexpr uint32_t cb_bias_cb = get_compile_time_arg_val(6);
    constexpr auto kv_args = TensorAccessorArgs<7>();
    constexpr auto gate_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();
    constexpr auto predecessor_kv_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr auto predecessor_score_args = TensorAccessorArgs<predecessor_kv_args.next_compile_time_args_offset()>();

    const auto kv = TensorAccessor(kv_args, kv_addr);
    const auto gate = TensorAccessor(gate_args, gate_addr);
    const auto bias = TensorAccessor(bias_args, bias_addr);
    const auto predecessor_kv = TensorAccessor(predecessor_kv_args, predecessor_kv_addr);
    const auto predecessor_score = TensorAccessor(predecessor_score_args, predecessor_score_addr);

    CircularBuffer candidate_kv(candidate_kv_cb);
    CircularBuffer candidate_score(candidate_score_cb);
    CircularBuffer ca_bias(ca_bias_cb);
    CircularBuffer cb_bias(cb_bias_cb);
    CircularBuffer scratch(scratch_cb);
    scratch.reserve_back(kScratchTiles);
    volatile tt_l1_ptr uint16_t* source = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch.get_write_ptr());
    Noc noc;

    // Candidate tile c, row r holds candidate c of window (32 * output_tile_row + r), which turns the
    // softmax over the 8 candidates into elementwise tile math downstream. Building that by walking
    // (window, candidate) re-reads the same source tile up to 256 times to keep 32 values out of each,
    // so this walks source tiles instead and scatters every row of one to the candidates it feeds.
    //
    // Source tile row j, local row i, holds token 32j + i. As a Cb candidate it feeds its own window
    // (8j + i/4); as a Ca candidate it feeds the NEXT one. Both halves therefore land at candidate
    // (half * 4 + i % 4) and row (window - 32 * output_tile_row) of that candidate's tile, and j only
    // has to sweep the four (Ca: five) source tile rows whose windows fall inside this output tile.
    const uint32_t output_tile_end = first_output_tile + output_tiles;
    for (uint32_t output_tile = first_output_tile; output_tile < output_tile_end; ++output_tile) {
        candidate_kv.reserve_back(8);
        candidate_score.reserve_back(8);
        ca_bias.reserve_back(1);
        cb_bias.reserve_back(1);
        volatile tt_l1_ptr uint16_t* candidate_kv_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(candidate_kv.get_write_ptr());
        volatile tt_l1_ptr uint16_t* candidate_score_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(candidate_score.get_write_ptr());
        for (uint32_t i = 0; i < 8 * kTileElems; ++i) {
            candidate_kv_ptr[i] = 0;
            candidate_score_ptr[i] = 0;
        }

        const uint32_t output_tile_row = output_tile / output_width_tiles;
        const uint32_t feature_tile = output_tile % output_width_tiles;
        const uint32_t window_begin = 32 * output_tile_row;

        // Every row of a candidate tile carries the same slot, so its whole bias is one row of the
        // bias tensor and the compute kernel can broadcast it. Ca reads the left half, Cb the right.
        noc.async_read(bias, ca_bias, kTileBytes, {.page_id = feature_tile}, {.offset_bytes = 0});
        noc.async_read(bias, cb_bias, kTileBytes, {.page_id = output_width_tiles + feature_tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        invalidate_l1_cache();

        // Windows past the chunk's real length stay zero, which is the pooled value they must produce.
        if (window_begin < complete_windows) {
            const uint32_t source_row_begin = 4 * output_tile_row;

            for (uint32_t half = 0; half < 2; ++half) {
                // Ca is the left half of the projection and feeds the window after its own.
                const uint32_t candidate_base = 4 * half;
                const uint32_t window_offset = half == 0 ? 1 : 0;
                const uint32_t column_tile = half * output_width_tiles + feature_tile;
                const uint32_t first_source_row =
                    (half == 0 && output_tile_row > 0) ? source_row_begin - 1 : source_row_begin;

                for (uint32_t j = first_source_row; j <= source_row_begin + 3; ++j) {
                    // Windows grow with j, so once this tile row starts past the end so does every
                    // later one. Bounding the read this way also keeps it inside the padded slab.
                    if (8 * j + window_offset >= complete_windows) {
                        break;
                    }
                    const uint32_t page = j * input_width_tiles + column_tile;
                    noc.async_read(
                        kv, scratch, kTileBytes, {.page_id = page}, {.offset_bytes = kScratchKv * kTileBytes});
                    noc.async_read(
                        gate, scratch, kTileBytes, {.page_id = page}, {.offset_bytes = kScratchGate * kTileBytes});
                    noc.async_read_barrier();
                    invalidate_l1_cache();

                    for (uint32_t i = 0; i < 32; ++i) {
                        const uint32_t window = 8 * j + i / 4 + window_offset;
                        if (window < window_begin || window >= window_begin + 32 || window >= complete_windows) {
                            continue;
                        }
                        const uint32_t candidate = candidate_base + (i & 3);
                        const uint32_t output_row = window - window_begin;
                        copy_tile_row(
                            candidate_kv_ptr + candidate * kTileElems, source + kScratchKv * kTileElems, output_row, i);
                        copy_tile_row(
                            candidate_score_ptr + candidate * kTileElems,
                            source + kScratchGate * kTileElems,
                            output_row,
                            i);
                    }
                }
            }

            // Window 0 has no predecessor window on this chip, so its Ca candidates come from the
            // state the exchange handed us. Those scores already carry the position bias, and the
            // compute kernel is about to broadcast it over the whole tile, so take it back out here.
            // Only these four rows pay for the round trip.
            if (output_tile_row == 0) {
                const uint32_t parity = (absolute_start / 4) & 1;
                const uint32_t page = parity * output_width_tiles + feature_tile;
                noc.async_read(
                    predecessor_kv,
                    scratch,
                    kTileBytes,
                    {.page_id = page},
                    {.offset_bytes = kScratchPredecessorKv * kTileBytes});
                noc.async_read(
                    predecessor_score,
                    scratch,
                    kTileBytes,
                    {.page_id = page},
                    {.offset_bytes = kScratchPredecessorScore * kTileBytes});
                noc.async_read_barrier();
                invalidate_l1_cache();

                volatile tt_l1_ptr uint16_t* ca_bias_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ca_bias.get_write_ptr());
                for (uint32_t slot = 0; slot < 4; ++slot) {
                    copy_tile_row(
                        candidate_kv_ptr + slot * kTileElems, source + kScratchPredecessorKv * kTileElems, 0, slot);
                    for (uint32_t col = 0; col < 32; ++col) {
                        // The state row and the bias row are both `slot`, so one offset serves both.
                        const uint32_t offset = tile_offset(slot, col);
                        const float unbiased = bf16_to_float(source[kScratchPredecessorScore * kTileElems + offset]) -
                                               bf16_to_float(ca_bias_ptr[offset]);
                        candidate_score_ptr[slot * kTileElems + tile_offset(0, col)] = float_to_bf16_rne(unbiased);
                    }
                }
            }
        }

        candidate_kv.push_back(8);
        candidate_score.push_back(8);
        ca_bias.push_back(1);
        cb_bias.push_back(1);
    }
    scratch.push_back(kScratchTiles);
}
