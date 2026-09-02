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
constexpr uint32_t kInputWidthTiles = 32;
constexpr uint32_t kStateWidthTiles = 16;

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
    const uint32_t base_kv_addr = get_arg_val<uint32_t>(3);
    const uint32_t base_score_addr = get_arg_val<uint32_t>(4);
    const uint32_t output_kv_addr = get_arg_val<uint32_t>(5);
    const uint32_t output_score_addr = get_arg_val<uint32_t>(6);
    const uint32_t local_valid = get_arg_val<uint32_t>(7);
    const uint32_t absolute_start = get_arg_val<uint32_t>(8);

    constexpr auto kv_args = TensorAccessorArgs<0>();
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
    scratch.reserve_back(7);
    volatile tt_l1_ptr uint16_t* memory = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch.get_write_ptr());
    Noc noc;

    for (uint32_t tile = 0; tile < 2 * kStateWidthTiles; ++tile) {
        noc.async_read(base_kv, scratch, kTileBytes, {.page_id = tile}, {.offset_bytes = 0});
        noc.async_read(base_score, scratch, kTileBytes, {.page_id = tile}, {.offset_bytes = kTileBytes});
        noc.async_read_barrier();
        invalidate_l1_cache();

        const uint32_t tile_row = tile / kStateWidthTiles;
        const uint32_t feature_tile = tile % kStateWidthTiles;
        for (uint32_t row_in_tile = 0; row_in_tile < 32; ++row_in_tile) {
            const uint32_t state_row = tile_row * 32 + row_in_tile;
            bool is_ca = false;
            uint32_t parity = 0;
            uint32_t slot = 0;
            if (state_row < 4) {
                is_ca = true;
                parity = 1;
                slot = state_row;
            } else if (state_row >= 32 && state_row < 36) {
                is_ca = true;
                parity = 0;
                slot = state_row - 32;
            } else if (state_row >= 4 && state_row < 8) {
                parity = 0;
                slot = state_row - 4;
            } else if (state_row >= 36 && state_row < 40) {
                parity = 1;
                slot = state_row - 36;
            } else {
                continue;
            }

            int32_t source_token = static_cast<int32_t>(local_valid) - 1;
            while (source_token >= 0) {
                const uint32_t position = absolute_start + static_cast<uint32_t>(source_token);
                if (position % 4 == slot && ((position / 4) & 1) == parity) {
                    break;
                }
                --source_token;
            }
            if (source_token < 0) {
                continue;
            }

            const uint32_t half = is_ca ? 0 : 1;
            const uint32_t source_col = half * 512 + feature_tile * 32;
            const uint32_t source_tile =
                (static_cast<uint32_t>(source_token) / 32) * kInputWidthTiles + source_col / 32;
            const uint32_t bias_tile = source_col / 32;
            noc.async_read(kv, scratch, kTileBytes, {.page_id = source_tile}, {.offset_bytes = 2 * kTileBytes});
            noc.async_read(gate, scratch, kTileBytes, {.page_id = source_tile}, {.offset_bytes = 3 * kTileBytes});
            noc.async_read(bias, scratch, kTileBytes, {.page_id = bias_tile}, {.offset_bytes = 4 * kTileBytes});
            noc.async_read_barrier();
            invalidate_l1_cache();

            const uint32_t source_row = static_cast<uint32_t>(source_token) % 32;
            for (uint32_t col = 0; col < 32; ++col) {
                const uint32_t dst = tile_offset(row_in_tile, col);
                const uint32_t src = tile_offset(source_row, col);
                const uint32_t bias_src = tile_offset(slot, col);
                memory[dst] = memory[2 * kTileElems + src];
                const float score =
                    bf16_to_float(memory[3 * kTileElems + src]) + bf16_to_float(memory[4 * kTileElems + bias_src]);
                memory[kTileElems + dst] = float_to_bf16_rne(score);
            }
        }

        noc.async_write(
            use<CircularBuffer::AddrSelector::WRITE_PTR>(scratch),
            output_kv,
            kTileBytes,
            {.offset_bytes = 0},
            {.page_id = tile});
        noc.async_write(
            use<CircularBuffer::AddrSelector::WRITE_PTR>(scratch),
            output_score,
            kTileBytes,
            {.offset_bytes = kTileBytes},
            {.page_id = tile});
        noc.async_write_barrier();
    }
    scratch.push_back(7);
}
