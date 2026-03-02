// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#ifndef REDUCE_ROW_SUM_VIA_MM
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#else
#include "ttnn/kernel/dataflow/generate_mm_scaler.hpp"
#endif

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    constexpr uint32_t scaler = get_compile_time_arg_val(0);
    constexpr auto tensor_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_id_in2 = 2;
#ifndef REDUCE_ROW_SUM_VIA_MM
    generate_reduce_scaler(cb_id_in2, scaler);
#else
    generate_mm_scaler(cb_id_in2, scaler);
#endif

    constexpr uint32_t cb_id_in0 = 0;

    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    auto tensor_accessor = TensorAccessor(tensor_args, src_addr, tile_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb_in0(cb_id_in0);

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        cb_in0.reserve_back(onetile);
        noc.async_read(tensor_accessor, cb_in0, tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in0.push_back(onetile);
    }
}
