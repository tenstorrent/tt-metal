// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.unpack_fp8_rows (fp8_rows.py), writer (NoC 0): each band's 4 bfloat8_b tiles to pages
// out_page0 + 4 b + t of the [.., M, 128] TILE output. common: out address. per core (value slots): out_page0, nbands
// (the band split as the reader's).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t GX = get_compile_time_arg_val(1);
    constexpr uint32_t NCORES = get_compile_time_arg_val(2);
    constexpr auto o_args = TensorAccessorArgs<3>();
    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto out = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), tile_bytes);
#ifdef UNPACK_COMMON_VALUES  // (option common_values: common args 1, 2)
    const uint32_t out_page0 = get_common_arg_val<uint32_t>(1);
    const uint32_t nbands = get_common_arg_val<uint32_t>(2);
#else
    const uint32_t out_page0 = get_arg_val<uint32_t>(0);
    const uint32_t nbands = get_arg_val<uint32_t>(1);
#endif
    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t per = nbands / NCORES, extra = nbands % NCORES;
    const uint32_t count = per + (c < extra ? 1u : 0u);
    const uint32_t first = c * per + (c < extra ? c : extra);
    for (uint32_t b = first; b < first + count; ++b) {
        cb_wait_front(cb_out, 4);
        uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < 4; ++t, l1 += tile_bytes) {
            noc_async_write(l1, out.get_noc_addr(out_page0 + 4 * b + t), tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 4);
    }
}
