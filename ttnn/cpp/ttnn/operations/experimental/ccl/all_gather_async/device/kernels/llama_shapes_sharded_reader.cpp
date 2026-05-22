// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include <tt-metalium/buffer_types.hpp>
#include <cstdint>
#include <utility>

// Legacy primitive(s) retained (#45003 item 4):
//   noc_async_read / noc_async_read_barrier paired with precomposed uint64_t shard noc addresses
//   computed via get_noc_addr(x, y, base) — Device 2.0 wrappers target endpoint objects, not raw addresses.

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t cb0_id = get_compile_time_arg_val(1);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(2);

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_core_tile_start_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;

    // interleaved addrgen

    CircularBuffer cb0(cb0_id);

    uint32_t tiles_read = 0;
    uint32_t shard_tile_id = first_core_tile_start_offset;
    uint32_t core_id = 0;
    while (tiles_read < num_tiles_to_read) {
        uint32_t num_tiles_to_read_this_core =
            std::min(num_tiles_per_core - shard_tile_id, num_tiles_to_read - tiles_read);
        cb0.reserve_back(num_tiles_to_read_this_core);
        const uint32_t l1_write_addr = cb0.get_write_ptr();
        uint64_t read_addr = get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], tensor_address0);
        read_addr += shard_tile_id * tensor0_page_size;

        noc_async_read(read_addr, l1_write_addr, num_tiles_to_read_this_core * tensor0_page_size);
        noc_async_read_barrier();

        cb0.push_back(num_tiles_to_read_this_core);
        tiles_read += num_tiles_to_read_this_core;
        shard_tile_id = 0;
        core_id++;
    }
}
