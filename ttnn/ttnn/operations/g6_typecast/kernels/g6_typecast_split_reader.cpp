// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast TILE sub-tile split reader (Perf 1, perf_experiments/subtile_split; NCRISC / NoC0): this core owns ONE
// unit = NF contiguous faces [first_face, first_face + NF) of tile `tile`. ONE single-packet read of unit_bytes from
// the tile's byte offset into the (tiny-tile) CB page. Input is never a block format, so a unit is one contiguous byte
// run. bf8b output with W % 16 != 0: zero the padding columns (col >= col0) of the local faces named by mask_bits
// (host: the unit's faces among {face_a, face_a + 2} of a last-column tile), as the op's reader does.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "g6_typecast_zones.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in_page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t unit_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t col0 = get_compile_time_arg_val(4);
    constexpr auto input_args = TensorAccessorArgs<5>();

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tile = get_arg_val<uint32_t>(0);
    const uint32_t off = get_arg_val<uint32_t>(1);
    const uint32_t mask_bits = get_arg_val<uint32_t>(2);

    const auto acc = TensorAccessor(input_args, input_addr, in_page_bytes);
    G6_ZONE("READ_UNIT");
    cb_reserve_back(cb_input_tiles, 1);
    const uint32_t l1 = get_write_ptr(cb_input_tiles);
    noc_async_read_one_packet(acc.get_noc_addr(tile, off), l1, unit_bytes);
    noc_async_read_barrier();
    if constexpr (col0 != 0) {
        for (uint32_t f = 0; f < 4; ++f) {
            if (mask_bits & (1u << f)) {
                volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1 + f * face_bytes);
                for (uint32_t r = 0; r < 16; ++r) {
                    for (uint32_t c = col0; c < 16; ++c) {
                        p[r * 16 + c] = 0;
                    }
                }
            }
        }
    }
    cb_push_back(cb_input_tiles, 1);
}
