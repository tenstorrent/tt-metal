// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Special case writer for unpad width 16 tensors
// Skip untilize and just copy f0 and f2 from input tiles to output tiles
void kernel_main() {
    uint32_t num_unpadded_output_rows = get_arg(args::num_unpadded_output_rows);
    uint32_t num_padded_tiles_per_core = get_arg(args::num_padded_tiles_per_core);

    // Output tile size in bytes (legacy: get_tile_size(cb_id_out)). Supplied as a named CTA so it
    // stays a compile-time constant usable in the static_assert below.
    constexpr uint32_t tile_size_in_bytes = get_arg(args::tile_size_in_bytes);
    constexpr uint32_t quarter_tile_size_in_bytes = tile_size_in_bytes / 4;

    const uint32_t batches_of_8 = num_padded_tiles_per_core / 8;
    const uint32_t remaining_tiles = num_padded_tiles_per_core % 8;

    Noc noc;
    DataflowBuffer cb_untilize_out(dfb::out);
    DataflowBuffer cb_out(dfb::out_sharded);

    cb_out.reserve_back(num_unpadded_output_rows);
    uint32_t l1_write_addr = cb_out.get_write_ptr();

    static_assert(quarter_tile_size_in_bytes <= NOC_MAX_BURST_SIZE);
    // set_state uses just x/y from the noc addr, addr is ignored
    noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
        UnicastEndpoint{},
        quarter_tile_size_in_bytes,
        {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = l1_write_addr});

    for (uint32_t i = 0; i < batches_of_8; i++) {
        cb_untilize_out.wait_front(8);
        uint32_t src_addr = cb_untilize_out.get_read_ptr();

        for (uint32_t j = 0; j < 8; j++) {
            CoreLocalMem<uint32_t> dst0(l1_write_addr);
            noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                UnicastEndpoint{},
                dst0,
                quarter_tile_size_in_bytes,
                {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                 .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                 .addr = src_addr},
                {.offset_bytes = 0});
            src_addr += 2 * quarter_tile_size_in_bytes;
            l1_write_addr += quarter_tile_size_in_bytes;

            CoreLocalMem<uint32_t> dst1(l1_write_addr);
            noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                UnicastEndpoint{},
                dst1,
                quarter_tile_size_in_bytes,
                {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                 .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                 .addr = src_addr},
                {.offset_bytes = 0});
            src_addr += 2 * quarter_tile_size_in_bytes;
            l1_write_addr += quarter_tile_size_in_bytes;
        }

        noc.async_read_barrier();
        cb_untilize_out.pop_front(8);
    }

    cb_untilize_out.wait_front(remaining_tiles);
    uint32_t src_addr = cb_untilize_out.get_read_ptr();
    for (uint32_t i = 0; i < remaining_tiles; i++) {
        CoreLocalMem<uint32_t> dst0(l1_write_addr);
        noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{},
            dst0,
            quarter_tile_size_in_bytes,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = src_addr},
            {.offset_bytes = 0});
        src_addr += 2 * quarter_tile_size_in_bytes;
        l1_write_addr += quarter_tile_size_in_bytes;

        CoreLocalMem<uint32_t> dst1(l1_write_addr);
        noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{},
            dst1,
            quarter_tile_size_in_bytes,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = src_addr},
            {.offset_bytes = 0});
        src_addr += 2 * quarter_tile_size_in_bytes;
        l1_write_addr += quarter_tile_size_in_bytes;
    }
    noc.async_read_barrier();
    cb_untilize_out.pop_front(remaining_tiles);

    cb_out.push_back(num_unpadded_output_rows);
}
