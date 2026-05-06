// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include "llk_defs.h"
#include <tt-metalium/constants.hpp>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// Row-major input pages (one page = one full tensor row along W). RM circular buffer index must match host factory.
void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_pages = get_arg_val<uint32_t>(1);
    uint32_t start_page = get_arg_val<uint32_t>(2);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(0);
    constexpr uint32_t W_logical = get_compile_time_arg_val(1);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(2);
    constexpr auto tensor_args = TensorAccessorArgs<3>();

    constexpr uint32_t cb_id_scaler = tt::CBIndex::c_2;
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    // SUM/AVG (device path uses PoolType::SUM for mean) + REDUCE_ROW uses the matmul reduce path; the scaler's
    // partial-fill helper interprets this count as rows along column 0 (see reduce_helpers_dataflow.inl). Use full
    // TILE_WIDTH so column-0 is populated for all tile rows after we zero RM padding below. MAX uses the row-0
    // partial path and needs valid width within the tile (clamp W_logical to one tile).
    const uint32_t scaler_valid_for_reduce = []() -> uint32_t {
        if constexpr (REDUCE_OP == ckernel::PoolType::SUM) {
            return tt::constants::TILE_WIDTH;
        }
        return (W_logical < tt::constants::TILE_WIDTH) ? W_logical : tt::constants::TILE_WIDTH;
    }();
    dataflow_kernel_lib::prepare_reduce_scaler<cb_id_scaler, REDUCE_OP, REDUCE_DIM>(scaler_f, scaler_valid_for_reduce);

    constexpr uint32_t cb_id_rm = tt::CBIndex::c_24;
    constexpr uint32_t onepage = 1;

    const uint32_t page_bytes = get_local_cb_interface(cb_id_rm).fifo_page_size;
    const uint32_t valid_row_bytes = W_logical * elem_bytes;

    auto tensor_accessor = TensorAccessor(tensor_args, src_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb_rm(cb_id_rm);

    uint32_t end_page = start_page + num_pages;
    for (uint32_t page_id = start_page; page_id < end_page; page_id++) {
        cb_rm.reserve_back(onepage);
        // Read only logical row bytes from source, then zero-fill staging tail.
        // This prevents tilize from pulling data from the next logical row when W is not tile-aligned.
        noc.async_read(tensor_accessor, cb_rm, valid_row_bytes, {.page_id = page_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        // Staging pages are sized to cover all tiles consumed by tilize_block; zero beyond logical width.
        if (valid_row_bytes < page_bytes) {
            volatile tt_l1_ptr uint8_t* row_base = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(cb_rm.get_write_ptr());
            for (uint32_t b = valid_row_bytes; b < page_bytes; b++) {
                row_base[b] = 0;
            }
        }
        cb_rm.push_back(onepage);
    }
}
