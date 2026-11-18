// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Port probe: record 0 = bank tables; then, per bank, the TensorAccessor read of pages (b + 7 q0, b + 7 q0 + 7);
// then one record per candidate (bank, x, y, noc): the same bytes read through that explicit endpoint.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(0);
    constexpr uint32_t rec_bytes = get_compile_time_arg_val(1);
    constexpr auto w_args = TensorAccessorArgs<2>();
    constexpr auto o_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    uint32_t a = 0;
    const uint32_t w_addr = get_arg_val<uint32_t>(a++);
    const uint32_t o_addr = get_arg_val<uint32_t>(a++);
    const uint32_t num_banks = get_arg_val<uint32_t>(a++);
    const uint32_t q0 = get_arg_val<uint32_t>(a++);
    const uint32_t w_page = get_arg_val<uint32_t>(a++);
    const uint32_t num_cand = get_arg_val<uint32_t>(a++);

    const auto w = TensorAccessor(w_args, w_addr, w_page);
    const auto o = TensorAccessor(o_args, o_addr, rec_bytes);
    const uint32_t l1 = get_write_ptr(cb_scratch);
    auto* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);

    for (uint32_t i = 0; i < rec_bytes / 4; ++i) {
        p[i] = 0;
    }
    for (uint32_t b = 0; b < num_banks; ++b) {
        p[4 * b + 0] = dram_bank_to_noc_xy[0][b];
        p[4 * b + 1] = dram_bank_to_noc_xy[1][b];
        p[4 * b + 2] = tensor_prefetcher_signal_noc_xy[b];
        p[4 * b + 3] = bank_to_dram_offset[b];
    }
    p[100] = my_x[0];
    p[101] = my_y[0];
    uint32_t rec = 0;
    noc_async_write(l1, o.get_noc_addr(rec++), rec_bytes);
    noc_async_write_barrier();

    for (uint32_t b = 0; b < num_banks; ++b) {
        noc_async_read(w.get_noc_addr(b + num_banks * q0), l1, rec_bytes);
        noc_async_read_barrier();
        noc_async_write(l1, o.get_noc_addr(rec++), rec_bytes);
        noc_async_write_barrier();
    }

    for (uint32_t c = 0; c < num_cand; ++c) {
        const uint32_t b = get_arg_val<uint32_t>(a++);
        const uint32_t x = get_arg_val<uint32_t>(a++);
        const uint32_t y = get_arg_val<uint32_t>(a++);
        const uint32_t n = get_arg_val<uint32_t>(a++);
        for (uint32_t i = 0; i < rec_bytes / 4; ++i) {
            p[i] = 0xDEADBEEF;
        }
        const uint32_t local = static_cast<uint32_t>(w.get_noc_addr(b + num_banks * q0));
        noc_async_read(get_noc_addr(x, y, local, n), l1, rec_bytes, n);
        noc_async_read_barrier(n);
        noc_async_write(l1, o.get_noc_addr(rec++), rec_bytes);
        noc_async_write_barrier();
    }
}
