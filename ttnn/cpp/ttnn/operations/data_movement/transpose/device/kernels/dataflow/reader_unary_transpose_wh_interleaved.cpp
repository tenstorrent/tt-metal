// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t N = get_arg_val<uint32_t>(1);
    uint32_t Ht = get_arg_val<uint32_t>(2);
    uint32_t Wt = get_arg_val<uint32_t>(3);
    uint32_t HtWt = get_arg_val<uint32_t>(4);

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id_in0);

#ifdef REDUCE_SCALER
    constexpr uint32_t cb_in_2 = 2;
    constexpr uint32_t scaler = get_compile_time_arg_val(src_args.next_compile_time_args_offset());
    experimental::CircularBuffer cb_scaler(cb_in_2);
    cb_scaler.reserve_back(1);
    if (scaler != 0) {
        uint16_t u = uint16_t(scaler >> 16);
        auto ptr = reinterpret_cast<uint16_t*>(cb_scaler.get_write_ptr());
        for (int j = 0; j < 1024; j++) {
            ptr[j] = uint16_t(0);
        }

        for (int k = 0; k < 4; k++) {
            for (int j = 0; j < 16; j++) {
                ptr[k * 256 + j] = u;
            }
        }
    }
    cb_scaler.push_back(1);
#endif

    uint32_t i_tile_N = 0;  // first tile in current batch
    uint32_t i_tile = 0;

    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const auto s = TensorAccessor(src_args, src_addr);

    // this reader will read a NHW tensor in NWH order
    for (uint32_t n = 0; n < N; n++) {
        i_tile = i_tile_N;
        for (uint32_t w = 0; w < Wt; w++) {
            for (uint32_t h = 0; h < Ht; h++) {
                cb.reserve_back(onetile);
                noc.async_read(s, cb, tile_bytes, {.page_id = i_tile}, {.offset_bytes = 0});
                noc.async_read_barrier();

                cb.push_back(onetile);
                i_tile += Wt;  // stride in H
            }  // Ht
            i_tile -= HtWt;  // go back to H=0
            i_tile += 1;     // increment Wt
        }  // Wt
        i_tile_N += HtWt;  // stride in batch/channel
    }  // N
}
