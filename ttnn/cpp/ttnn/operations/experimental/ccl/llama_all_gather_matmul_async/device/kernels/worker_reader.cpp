// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include <cstdint>
#include <utility>

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
    address_t intermediate_tensor_address0 = get_arg_val<address_t>(arg_idx++);
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_core_tile_start_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t intermediate_first_core_tile_start_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_index = get_arg_val<uint32_t>(arg_idx++);
    const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    uint32_t num_cores_to_write_to = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* core_noc_x_to_write_to = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores_to_write_to;
    tt_l1_ptr uint32_t* core_noc_y_to_write_to = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores_to_write_to;

    // interleaved addrgen

    Noc noc_obj;
    CircularBuffer cb0(cb0_id);

    uint32_t tiles_read = 0;
    uint32_t shard_tile_id = first_core_tile_start_offset;
    uint32_t core_id = 0;
    cb0.reserve_back(num_tiles_to_read);
    uint32_t l1_write_addr = cb0.get_write_ptr();
    while (tiles_read < num_tiles_to_read) {
        uint32_t num_tiles_to_read_this_core =
            std::min(num_tiles_per_core - shard_tile_id, num_tiles_to_read - tiles_read);
        // cb_reserve_back(cb0_id, num_tiles_to_read_this_core);
        // const uint32_t l1_write_addr = get_write_ptr(cb0_id);
        const uint32_t shard_addr = tensor_address0 + shard_tile_id * tensor0_page_size;

        noc_obj.async_read(
            UnicastEndpoint{},
            CoreLocalMem<uint8_t>(l1_write_addr),
            num_tiles_to_read_this_core * tensor0_page_size,
            {.noc_x = core_noc_x[core_id], .noc_y = core_noc_y[core_id], .addr = shard_addr},
            {});
        // noc_async_read_barrier();

        // cb_push_back(cb0_id, num_tiles_to_read_this_core);
        l1_write_addr += num_tiles_to_read_this_core * tensor0_page_size;
        tiles_read += num_tiles_to_read_this_core;
        shard_tile_id = 0;
        core_id++;
    }
    noc_obj.async_read_barrier();
    cb0.push_back(num_tiles_to_read);
    core_id = 0;
    const uint32_t intermediate_dest_addr =
        intermediate_tensor_address0 + intermediate_first_core_tile_start_offset * tensor0_page_size;
    uint32_t l1_read_addr = cb0.get_read_ptr();
    noc_obj.async_write(
        CoreLocalMem<uint8_t>(l1_read_addr),
        UnicastEndpoint{},
        num_tiles_to_read * tensor0_page_size,
        {},
        {.noc_x = core_noc_x_to_write_to[core_id],
         .noc_y = core_noc_y_to_write_to[core_id],
         .addr = intermediate_dest_addr});

    // notify local receiver core
    uint64_t out_ready_sem_noc_addr =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
    noc_semaphore_inc(out_ready_sem_noc_addr, 1);
    noc_obj.async_write_barrier();
    noc_obj.async_atomic_barrier();
}
