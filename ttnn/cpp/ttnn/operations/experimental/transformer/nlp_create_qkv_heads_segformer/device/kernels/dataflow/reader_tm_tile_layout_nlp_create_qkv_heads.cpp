// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

    // READER RUNTIME ARGS
    uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_tensor_addr = get_arg_val<uint32_t>(1);
    uint32_t num_blocks = get_arg_val<uint32_t>(2);
    uint32_t in0_tensor_tile_id = get_arg_val<uint32_t>(3);
    uint32_t in1_tensor_tile_id = get_arg_val<uint32_t>(4);

    // COMPILE TIME ARGS
    // interleaved accessor args
    // READER COMPILE TIME ARGS
    constexpr uint32_t q_num_tiles = get_compile_time_arg_val(0);
    constexpr auto in0_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_id_qv = 1;  // cb for Q, V heads

    constexpr uint32_t onetile = 1;
    const DataFormat data_format = get_dataformat(cb_id_qv);

    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr);

    CircularBuffer cb_qv(cb_id_qv);
    uint32_t tile_bytes = get_tile_size(cb_id_qv);

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Q
        for (uint32_t i = 0; i < q_num_tiles; i++) {
            cb_qv.reserve_back(onetile);
            uint32_t l1_write_addr = cb_qv.get_write_ptr();
            noc.async_read(s0, CoreLocalMem<uint32_t>(l1_write_addr), tile_bytes, {.page_id = in0_tensor_tile_id}, {});
            noc.async_read_barrier();
            cb_qv.push_back(onetile);
            in0_tensor_tile_id++;
        }
    }
}
