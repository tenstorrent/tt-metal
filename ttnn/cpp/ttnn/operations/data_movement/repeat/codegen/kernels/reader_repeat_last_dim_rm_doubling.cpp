// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_pages = get_arg_val<uint32_t>(1);
    uint32_t start_page = get_arg_val<uint32_t>(2);

    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t in_read_size = get_compile_time_arg_val(1);
    constexpr uint32_t out_l1_stride = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t NUM_REPEATS = get_compile_time_arg_val(4);
    constexpr uint32_t BATCH = get_compile_time_arg_val(5);
    constexpr auto src_args = TensorAccessorArgs<6>();
    constexpr uint32_t row_bytes = stick_size * NUM_REPEATS;
    constexpr uint32_t seed_bytes = 32;
    constexpr uint32_t fill_bytes = seed_bytes < row_bytes ? seed_bytes : row_bytes;
    static_assert(stick_size == 2 || stick_size == 4 || stick_size == 8);
    static_assert(row_bytes % 4 == 0);

    const auto s = TensorAccessor(src_args, src_addr);
    Noc noc;
    CircularBuffer cb_in(cb_id);
    UnicastEndpoint self_ep;
    const uint32_t my_noc_x = my_x[noc.get_noc_id()];
    const uint32_t my_noc_y = my_y[noc.get_noc_id()];

    uint32_t src_page = start_page;
    uint32_t pages_left = num_pages;
    while (pages_left > 0) {
        uint32_t batch = (pages_left < BATCH) ? pages_left : BATCH;
        cb_in.reserve_back(batch);
        uint32_t l1_base = cb_in.get_write_ptr();
        for (uint32_t t = 0; t < batch; t++) {
            noc.async_read(
                s, cb_in, in_read_size, {.page_id = src_page, .offset_bytes = 0}, {.offset_bytes = t * out_l1_stride});
            src_page++;
        }
        noc.async_read_barrier();

        for (uint32_t t = 0; t < batch; t++) {
            volatile tt_l1_ptr uint32_t* row =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_base + t * out_l1_stride);
            uint32_t w0, w1;
            if constexpr (stick_size == 2) {
                const uint32_t v = row[0] & 0xFFFFu;
                w0 = v | (v << 16);
                w1 = w0;
            } else if constexpr (stick_size == 4) {
                w0 = row[0];
                w1 = w0;
            } else {
                w0 = row[0];
                w1 = row[1];
            }
            constexpr uint32_t words = fill_bytes / 4;
            for (uint32_t i = 0; i < words; i += 2) {
                row[i] = w0;
                if (i + 1 < words) {
                    row[i + 1] = w1;
                }
            }
        }
        asm volatile("" ::: "memory");

        for (uint32_t len = seed_bytes; len < row_bytes; len *= 2) {
            const uint32_t n = (2 * len <= row_bytes) ? len : (row_bytes - len);
            for (uint32_t t = 0; t < batch; t++) {
                const uint32_t l1_addr = l1_base + t * out_l1_stride;
                noc.async_read(
                    self_ep,
                    cb_in,
                    n,
                    {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = l1_addr},
                    {.offset_bytes = t * out_l1_stride + len});
            }
            noc.async_read_barrier();
        }

        cb_in.push_back(batch);
        pages_left -= batch;
    }
}
