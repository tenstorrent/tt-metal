// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

using namespace tt;

void kernel_main() {
    uint32_t start_id = get_arg(args::start_id);
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t end_id = start_id + num_tiles;

    const auto output_addrg = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb_intermed(dfb::intermed);
    DataflowBuffer dfb_dst(dfb::dst);

    const uint32_t page_bytes = dfb_dst.get_entry_size();

    dfb_dst.reserve_back(1);
    uint32_t dst_dfb_write_ptr = dfb_dst.get_write_ptr();

    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_intermed.wait_front(1);

        uint32_t intermed_dfb_read_ptr = dfb_intermed.get_read_ptr();
        auto intermed_dfb_addr = reinterpret_cast<float*>(intermed_dfb_read_ptr);

#ifdef OUTPUT_DTYPE_FLOAT32
        noc.async_write(CoreLocalMem<uint32_t>(intermed_dfb_read_ptr), output_addrg, page_bytes, {}, {.page_id = i});
        noc.async_write_barrier();
        dfb_intermed.pop_front(1);
#endif

#ifdef OUTPUT_DTYPE_BFLOAT16
        auto dst_dfb_addr = reinterpret_cast<uint8_t*>(dst_dfb_write_ptr);
        for (uint32_t k = 0; k < constants::TILE_WIDTH; k++) {
            for (uint32_t j = 0; j < constants::TILE_HEIGHT; j++) {
                float rand_float = *intermed_dfb_addr;

                uint16_t* uint16_ptr = reinterpret_cast<uint16_t*>(&rand_float) + 1;
                *(uint16_t*)dst_dfb_addr = *uint16_ptr;
                dst_dfb_addr += 2;
                intermed_dfb_addr += 1;
            }
        }
        dfb_intermed.pop_front(1);

        noc.async_write(CoreLocalMem<uint32_t>(dst_dfb_write_ptr), output_addrg, page_bytes, {}, {.page_id = i});
        noc.async_write_barrier();
#endif
    }

    // dst DFB is reserved once as a conversion-staging region (consumed only by direct NOC
    // writes, never streamed to a consumer); commit the reservation so the DFB is left balanced.
    dfb_dst.push_back(1);
}
