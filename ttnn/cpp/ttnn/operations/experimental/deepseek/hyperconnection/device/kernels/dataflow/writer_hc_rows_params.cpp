// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {

FORCE_INLINE uint32_t tile_face_index(uint32_t r, uint32_t c) {
    const uint32_t face = ((r >= 16) ? 2u : 0u) + ((c >= 16) ? 1u : 0u);
    return face * 256u + (r & 15u) * 16u + (c & 15u);
}

}  // namespace

// The post / comb cores' writer for a block of rows rows: the compute leaves every row's values in one 32x32 tile
// (hc_post_pre_rows, hc_comb_sinkhorn_rows), which this spreads into one zero-padded tile per row of the outputs:
//   role 1: post [1, rows, H, 1] (tile j's column 0) and pre [1, rows, 1, H] (tile j's row 0), from (row k, column j)
//           of faces 0 and 2;
//   role 2: comb [1, rows, H, H], (r, c) of row j from face row r + 4 (c / 2), face column 2 j + c % 2 of face 0.
//
// Compile-time args: 0: role   1: cb_in   2: cb_scratch (rows tiles per output)   3: rows   4: num_streams
//   5+: TensorAccessorArgs(out), TensorAccessorArgs(out2)
// Runtime args: 0: out address   1: out2 address (pre, role 1)
void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(2);
    constexpr uint32_t rows = get_compile_time_arg_val(3);
    constexpr uint32_t num_streams = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();
    constexpr auto out2_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    constexpr uint32_t num_outputs = role == 1 ? 2 : 1;
    constexpr uint32_t tile_bytes = 32 * 32 * sizeof(uint16_t);

    Noc noc;
    CircularBuffer scratch(cb_scratch);
    scratch.reserve_back(num_outputs * rows);
    noc.async_write_zeros(scratch, num_outputs * rows * tile_bytes, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();

    CircularBuffer in(cb_in);
    in.wait_front(1);
    const auto* src = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(in.get_read_ptr());
    auto* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch.get_write_ptr());
    constexpr uint32_t tile_elems = 32 * 32;
    for (uint32_t j = 0; j < rows; ++j) {
        volatile tt_l1_ptr uint16_t* t = dst + j * tile_elems;
        if constexpr (role == 1) {
            volatile tt_l1_ptr uint16_t* t2 = dst + (rows + j) * tile_elems;
            for (uint32_t k = 0; k < num_streams; ++k) {
                t[tile_face_index(k, 0)] = src[tile_face_index(k, j)];
                t2[tile_face_index(0, k)] = src[tile_face_index(16 + k, j)];
            }
        } else {
            for (uint32_t r = 0; r < num_streams; ++r) {
                for (uint32_t c = 0; c < num_streams; ++c) {
                    t[tile_face_index(r, c)] = src[tile_face_index(r + 4 * (c >> 1), 2 * j + (c & 1))];
                }
            }
        }
    }
    in.pop_front(1);

    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(0));
    for (uint32_t j = 0; j < rows; ++j) {
        noc.async_write(scratch, out, tile_bytes, {.offset_bytes = j * tile_bytes}, {.page_id = j});
    }
    if constexpr (role == 1) {
        const auto out2 = TensorAccessor(out2_args, get_arg_val<uint32_t>(1));
        for (uint32_t j = 0; j < rows; ++j) {
            noc.async_write(scratch, out2, tile_bytes, {.offset_bytes = (rows + j) * tile_bytes}, {.page_id = j});
        }
    }
    noc.async_write_barrier();
}
