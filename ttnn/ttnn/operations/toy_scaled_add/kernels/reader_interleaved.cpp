// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// toy_scaled_add reader for interleaved a / b (NCRISC, NoC0).
//
// When gamma is present it is loaded once, all Wt tiles of its row, before anything else: the
// compute kernel holds that row for the whole walk and reuses tile c for every row's column c.
// Then a and b stream tile by tile over this core's tile-rows, both reads of a tile in flight
// before one barrier, into double-buffered CBs.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/operations/toy_scaled_add/device/kernels/toy_scaled_add_args.hpp"

using namespace toy_scaled_add;

void kernel_main() {
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();

    const uint32_t row_start = get_arg_val<uint32_t>(core_arg::ROW_START);
    const uint32_t num_rows = get_arg_val<uint32_t>(core_arg::NUM_ROWS);

    Noc noc;
    CircularBuffer cb_a(cb::A);
    CircularBuffer cb_b(cb::B);
    const uint32_t a_tile_bytes = get_tile_size(cb::A);
    const uint32_t b_tile_bytes = get_tile_size(cb::B);
    const auto a = TensorAccessor(a_args, get_common_arg_val<uint32_t>(reader_arg::A_ADDR), a_tile_bytes);
    const auto b = TensorAccessor(b_args, get_common_arg_val<uint32_t>(reader_arg::B_ADDR), b_tile_bytes);

#ifdef TOY_SCALED_ADD_HAS_GAMMA
    CircularBuffer cb_gamma(cb::GAMMA);
    const uint32_t gamma_tile_bytes = get_tile_size(cb::GAMMA);
    const auto gamma =
        TensorAccessor(gamma_args, get_common_arg_val<uint32_t>(reader_arg::GAMMA_ADDR), gamma_tile_bytes);
    cb_gamma.reserve_back(Wt);
    for (uint32_t c = 0; c < Wt; ++c) {
        noc.async_read(gamma, cb_gamma, gamma_tile_bytes, {.page_id = c}, {.offset_bytes = c * gamma_tile_bytes});
    }
    noc.async_read_barrier();
    cb_gamma.push_back(Wt);
#endif

    const uint32_t tile_end = (row_start + num_rows) * Wt;
    for (uint32_t tile = row_start * Wt; tile < tile_end; ++tile) {
        cb_a.reserve_back(1);
        cb_b.reserve_back(1);
        noc.async_read(a, cb_a, a_tile_bytes, {.page_id = tile}, {});
        noc.async_read(b, cb_b, b_tile_bytes, {.page_id = tile}, {});
        noc.async_read_barrier();
        cb_a.push_back(1);
        cb_b.push_back(1);
    }
}
