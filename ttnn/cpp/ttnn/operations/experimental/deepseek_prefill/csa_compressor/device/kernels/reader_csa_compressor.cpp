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

    constexpr uint32_t candidate_kv_cb = get_compile_time_arg_val(0);
    constexpr uint32_t candidate_score_cb = get_compile_time_arg_val(1);
    constexpr uint32_t scratch_cb = get_compile_time_arg_val(2);
    constexpr uint32_t input_height_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t input_width_tiles = get_compile_time_arg_val(4);
    constexpr auto kv_args = TensorAccessorArgs<5>();
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
    CircularBuffer scratch(scratch_cb);
    scratch.reserve_back(5);
    Noc noc;

    constexpr uint32_t output_width_tiles = 16;
    for (uint32_t output_tile = 0; output_tile < output_tiles; ++output_tile) {
        candidate_kv.reserve_back(8);
        candidate_score.reserve_back(8);
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
        for (uint32_t output_row = 0; output_row < 32; ++output_row) {
            const uint32_t window = output_tile_row * 32 + output_row;
            if (window >= complete_windows) {
                continue;
            }
            for (uint32_t candidate = 0; candidate < 8; ++candidate) {
                const bool is_ca = candidate < 4;
                const uint32_t local_slot = candidate & 3;
                uint32_t kv_tile;
                uint32_t score_tile;
                uint32_t source_row;
                bool score_is_prebiased = false;
                uint32_t bias_slot = 0;

                if (is_ca && window == 0) {
                    const uint32_t parity = (absolute_start / 4) & 1;
                    const uint32_t state_row = parity * 32 + local_slot;
                    kv_tile = (state_row / 32) * output_width_tiles + feature_tile;
                    score_tile = kv_tile;
                    source_row = state_row % 32;
                    score_is_prebiased = true;
                    noc.async_read(predecessor_kv, scratch, kTileBytes, {.page_id = kv_tile}, {.offset_bytes = 0});
                    noc.async_read(
                        predecessor_score, scratch, kTileBytes, {.page_id = score_tile}, {.offset_bytes = kTileBytes});
                } else {
                    const uint32_t token = is_ca ? (window - 1) * 4 + local_slot : window * 4 + local_slot;
                    const uint32_t source_col = (is_ca ? 0 : 512) + feature_tile * 32;
                    kv_tile = (token / 32) * input_width_tiles + source_col / 32;
                    score_tile = kv_tile;
                    source_row = token % 32;
                    bias_slot = (absolute_start + token) % 4;
                    noc.async_read(kv, scratch, kTileBytes, {.page_id = kv_tile}, {.offset_bytes = 0});
                    noc.async_read(gate, scratch, kTileBytes, {.page_id = score_tile}, {.offset_bytes = kTileBytes});
                    noc.async_read(
                        bias, scratch, kTileBytes, {.page_id = source_col / 32}, {.offset_bytes = 2 * kTileBytes});
                }
                noc.async_read_barrier();
                invalidate_l1_cache();

                volatile tt_l1_ptr uint16_t* source =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch.get_write_ptr());
                for (uint32_t col = 0; col < 32; ++col) {
                    const uint32_t dst = candidate * kTileElems + tile_offset(output_row, col);
                    const uint32_t src = tile_offset(source_row, col);
                    candidate_kv_ptr[dst] = source[src];
                    if (score_is_prebiased) {
                        candidate_score_ptr[dst] = source[kTileElems + src];
                    } else {
                        const uint32_t bias_src = tile_offset(bias_slot, col);
                        const float biased_score =
                            bf16_to_float(source[kTileElems + src]) + bf16_to_float(source[2 * kTileElems + bias_src]);
                        candidate_score_ptr[dst] = float_to_bf16_rne(biased_score);
                    }
                }
            }
        }
        candidate_kv.push_back(8);
        candidate_score.push_back(8);
    }
    scratch.push_back(5);
}
