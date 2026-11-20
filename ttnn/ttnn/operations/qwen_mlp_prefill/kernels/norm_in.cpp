// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "named_args.hpp"

#include "reduce_scaler.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t tile_bytes = CT_ARG("tile_bytes_bf16");
    constexpr uint32_t row_tiles = CT_ARG("h_tiles");
    const auto h = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_h")>(), RT_ARG("h_addr"), tile_bytes);
    const auto w_norm = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_w_norm")>(), RT_ARG("w_norm_addr"), tile_bytes);
    CircularBuffer in(CT_ARG("cb_norm_in"));

    fill_reduce_scaler(noc, CT_ARG("cb_norm_scaler"), tile_bytes);
    fill_ones(noc, CT_ARG("cb_norm_ones"), tile_bytes);
    in.reserve_back(row_tiles);
    for (uint32_t t = 0; t < row_tiles; t++) {
        noc.async_read(
            w_norm, CoreLocalMem<uint32_t>(in.get_write_ptr() + t * tile_bytes), tile_bytes, {.page_id = t}, {});
    }
    noc.async_read_barrier();
    in.push_back(row_tiles);

    for (uint32_t m = RT_ARG("index"); m < CT_ARG("rows"); m += CT_ARG("norm_cores")) {
        in.reserve_back(row_tiles);
        for (uint32_t t = 0; t < row_tiles; t++) {
            noc.async_read(
                h,
                CoreLocalMem<uint32_t>(in.get_write_ptr() + t * tile_bytes),
                tile_bytes,
                {.page_id = m * row_tiles + t},
                {});
        }
        noc.async_read_barrier();
        in.push_back(row_tiles);
    }
}
