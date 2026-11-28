// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tensix_transpose experiment, writer. For unit (n, h, wb) the compute pushes one cb_rows page per
// c-tile ct: 32 rows c = ct * 32 + r of kwu * 32 values (row stride kwu * 64 B), row c holding
// out[n, c, h, wb * block_w + (0 .. width)]. Row c goes to output page (n * C + c) * H + h at byte
// offset wb * block_w * 2 (a multiple of 64 B: block_w is a multiple of 32 unless a unit is a whole
// row). Rows c >= C of the last c-tile are never sent.
// Raw dataflow: kernel_lib has no helper that writes strided partial rows to interleaved pages.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_rows = get_compile_time_arg_val(0);
constexpr uint32_t tensor_c = get_compile_time_arg_val(1);
constexpr uint32_t tensor_ct = get_compile_time_arg_val(2);
constexpr uint32_t tensor_h = get_compile_time_arg_val(3);
constexpr uint32_t units_per_row = get_compile_time_arg_val(4);
constexpr uint32_t block_w = get_compile_time_arg_val(5);
constexpr uint32_t tail_w = get_compile_time_arg_val(6);
constexpr uint32_t kw = get_compile_time_arg_val(7);
constexpr uint32_t kw_tail = get_compile_time_arg_val(8);
constexpr uint32_t out_page_bytes = get_compile_time_arg_val(9);
constexpr auto output_args = TensorAccessorArgs<10>();

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t value_bytes = 2;

#ifdef TTX_ZONES
#define TTX_ZONE(name) DeviceZoneScopedN(name)
#else
#define TTX_ZONE(name)
#endif

void kernel_main() {
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    uint32_t remaining = get_arg_val<uint32_t>(0);
    uint32_t wb = get_arg_val<uint32_t>(1);
    uint32_t h = get_arg_val<uint32_t>(2);
    uint32_t page = get_arg_val<uint32_t>(3);  // p_out(n, c = 0, h)

    const auto output = TensorAccessor(output_args, output_addr, out_page_bytes);

    while (remaining > 0) {
        const bool tail = wb == units_per_row - 1;
        const uint32_t bytes = (tail ? tail_w : block_w) * value_bytes;
        const uint32_t row_stride = (tail ? kw_tail : kw) * TILE_ROWS * value_bytes;
        const uint32_t offset = wb * block_w * value_bytes;
        for (uint32_t ct = 0; ct < tensor_ct; ++ct) {
            {
                TTX_ZONE("WR-WAIT");
                cb_wait_front(cb_rows, 1);
            }
            uint32_t src = get_read_ptr(cb_rows);
            const uint32_t c0 = ct * TILE_ROWS;
            const uint32_t rows = (tensor_c - c0) < TILE_ROWS ? (tensor_c - c0) : TILE_ROWS;
            {
                TTX_ZONE("WR-ISSUE");
                uint32_t out_page = page + c0 * tensor_h;
                for (uint32_t r = 0; r < rows; ++r) {
#ifndef TTX_NO_WRITES
                    noc_async_write<kw * TILE_ROWS * value_bytes>(src, output.get_noc_addr(out_page, offset), bytes);
#endif
                    src += row_stride;
                    out_page += tensor_h;
                }
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_rows, 1);
        }
        if (++wb == units_per_row) {
            wb = 0;
            ++page;
            if (++h == tensor_h) {
                h = 0;
                page += (tensor_c - 1) * tensor_h;
            }
        }
        --remaining;
    }
    noc_async_write_barrier();
}
