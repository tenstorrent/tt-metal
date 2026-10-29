// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The rounded bfp tiles [t0, t0 + n) written as pages page0 + t0 + t of the output (page0: the table's first page,
// one value for every core).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ckv_share.hpp"

void kernel_main() {
#ifdef CKV_COMMON_GX  // (option compact: common [out, per, extra, page0], no per-core args)
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    uint32_t t_first, n_tiles;
    ckv_share(get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(2), CKV_COMMON_GX, t_first, n_tiles);
    const uint32_t page0 = t_first + get_common_arg_val<uint32_t>(3);
#else
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t page0 = get_arg_val<uint32_t>(2) + get_arg_val<uint32_t>(3);
#endif

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);

    for (uint32_t t0 = 0; t0 < n_tiles; t0 += BLOCK) {
        const uint32_t nb = n_tiles - t0 < BLOCK ? n_tiles - t0 : BLOCK;
        cb_wait_front(cb_out, nb);
        uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < nb; ++t, l1 += tile_bytes) {
            noc_async_write(l1, out_acc.get_noc_addr(page0 + t0 + t), tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, nb);
    }
}
