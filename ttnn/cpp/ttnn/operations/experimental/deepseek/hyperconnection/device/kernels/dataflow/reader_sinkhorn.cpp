// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {

// Fill a single tile with the reduce scaler value (bf16 hi-half) in the standard layout
// expected by reduce_tile: the first column of each of the 4 faces.
FORCE_INLINE void generate_reduce_scaler(uint32_t cb_id, uint16_t scaler) {
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = 0;
    }
    if (scaler != 0) {
        for (uint32_t k = 0; k < 4; ++k) {
            const uint32_t idx = k << 8;
            for (uint32_t j = 0; j < 16; ++j) {
                ptr[idx + j] = scaler;
            }
        }
    }
    cb_push_back(cb_id, 1);
}

// Fill a single tile with 1.0 (bf16 hi-half) inside the valid HxH block and 0 elsewhere,
// honouring the 4x(16x16)-face tile layout. H <= 32.
FORCE_INLINE void generate_hxh_mask(uint32_t cb_id, uint32_t h, uint16_t one_bits) {
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = 0;
    }
    for (uint32_t r = 0; r < h; ++r) {
        for (uint32_t c = 0; c < h; ++c) {
            const uint32_t face = ((r >= 16) ? 2u : 0u) + ((c >= 16) ? 1u : 0u);
            const uint32_t idx = face * 256u + (r & 15u) * 16u + (c & 15u);
            ptr[idx] = one_bits;
        }
    }
    cb_push_back(cb_id, 1);
}

}  // namespace

void kernel_main() {
    const uint32_t comb_w_addr = get_arg_val<uint32_t>(0);
    const uint32_t comb_bias_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_comb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_comb_bias = get_compile_time_arg_val(1);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(4);
    constexpr uint32_t num_streams = get_compile_time_arg_val(5);

    constexpr auto comb_w_args = TensorAccessorArgs<6>();
    constexpr auto comb_bias_args = TensorAccessorArgs<comb_w_args.next_compile_time_args_offset()>();

    const auto comb_w = TensorAccessor(comb_w_args, comb_w_addr);
    const auto comb_bias = TensorAccessor(comb_bias_args, comb_bias_addr);

    // Reduce scaler is a constant 1.0 (plain sum / max); only the bf16 hi-half is used.
    generate_reduce_scaler(cb_scaler, static_cast<uint16_t>(scaler_bits >> 16));
    // Ones mask delimiting the valid HxH block (1.0 inside, 0 in the tile padding).
    generate_hxh_mask(cb_mask, num_streams, static_cast<uint16_t>(scaler_bits >> 16));

    Noc noc;
    CircularBuffer cb_w(cb_comb_w);
    CircularBuffer cb_b(cb_comb_bias);

    constexpr uint32_t one_tile = 1;
    cb_w.reserve_back(one_tile);
    cb_b.reserve_back(one_tile);

    noc.async_read(comb_w, cb_w, cb_w.get_tile_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(comb_bias, cb_b, cb_b.get_tile_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();

    cb_w.push_back(one_tile);
    cb_b.push_back(one_tile);
}
