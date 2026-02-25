// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Solution: Reader kernel with DPRINT debugging

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    const uint32_t tile_size = get_tile_size(cb_in);

    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto input = TensorAccessor(input_args, input_addr, tile_size);

    constexpr uint32_t num_tiles_offset = input_args.next_compile_time_args_offset();
    constexpr uint32_t num_tiles = get_compile_time_arg_val(num_tiles_offset);

    DPRINT << "Reader: num_tiles=" << num_tiles << " tile_size=" << tile_size << " input_addr=0x" << HEX() << input_addr
           << DEC() << ENDL();

    for (uint32_t i = 0; i < num_tiles; i++) {
        DPRINT << "Reader: tile " << i << "/" << num_tiles << ENDL();

        cb_reserve_back(cb_in, 1);
        uint32_t l1_addr = get_write_ptr(cb_in);
        DPRINT << "  l1_addr=0x" << HEX() << l1_addr << DEC() << ENDL();

        noc_async_read_tile(i, input, l1_addr);
        noc_async_read_barrier();

        SliceRange sr = SliceRange{.h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1};
        DPRINT << "  tile values (4x4):" << ENDL();
        DPRINT << TileSlice(cb_in, 0, sr, TSLICE_INPUT_CB, TSLICE_WR_PTR, true, true) << ENDL();

        cb_push_back(cb_in, 1);
    }

    DPRINT << "Reader: done" << ENDL();
}
