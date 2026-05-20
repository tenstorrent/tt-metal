// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(0);
    constexpr auto tensor_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_id_in2 = 2;
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_id_in2, REDUCE_OP, REDUCE_DIM>(scaler_f);

    constexpr uint32_t cb_id_in0 = 0;
    // Welford-fp32 alias: shares cb_in0's memory but has independent rd/wr FIFO pointers,
    // so the reader must reserve_back / push_back the alias every iteration to keep its
    // pointers in lock-step with cb_in0. Without this the alias rd_ptr stays frozen at its
    // initial position; on the second NCHt iteration the compute kernel would read stale
    // (already-popped) data via the alias. Mirrors layernorm's reader pattern.
    constexpr uint32_t cb_id_in_welford_alias = get_named_compile_time_arg_val("cb_in_welford_alias");
    constexpr bool welford_fp32_alias = get_named_compile_time_arg_val("welford_fp32_alias") != 0;

    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    auto tensor_accessor = TensorAccessor(tensor_args, src_addr);

    Noc noc;
    CircularBuffer cb_in0(cb_id_in0);
    CircularBuffer cb_in_welford_alias(cb_id_in_welford_alias);

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        cb_in0.reserve_back(onetile);
        noc.async_read(tensor_accessor, cb_in0, tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in0.push_back(onetile);
        if constexpr (welford_fp32_alias) {
            cb_in_welford_alias.reserve_back(onetile);
            cb_in_welford_alias.push_back(onetile);
        }
    }
}
