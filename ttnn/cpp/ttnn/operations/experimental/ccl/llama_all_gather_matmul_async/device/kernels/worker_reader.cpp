// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include <cstdint>
#include <utility>
#include "tools/profiler/kernel_profiler.hpp"

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
    DPRINT << "CCL WRD num_tiles_to_read: " << num_tiles_to_read << ENDL();
    DPRINT << "CCL WRD intermediate_first_core_tile_start_offset: " << intermediate_first_core_tile_start_offset
           << ENDL();
    DPRINT << "CCL WRD tensor0_page_size: " << tensor0_page_size << ENDL();
    // interleaved addrgen
    uint32_t tiles_read = 0;
    uint32_t shard_tile_id = first_core_tile_start_offset;
    uint32_t core_id = 0;
    cb_reserve_back(cb0_id, num_tiles_to_read);
    uint32_t l1_write_addr = get_write_ptr(cb0_id);

    // comment this out
    while (tiles_read < num_tiles_to_read) {
        DPRINT << "CCL WRD core_noc_x[core_id]: " << core_noc_x[core_id] << ENDL();
        DPRINT << "CCL WRD core_noc_y[core_id]: " << core_noc_y[core_id] << ENDL();
        uint32_t num_tiles_to_read_this_core =
            std::min(num_tiles_per_core - shard_tile_id, num_tiles_to_read - tiles_read);
        // cb_reserve_back(cb0_id, num_tiles_to_read_this_core);
        // const uint32_t l1_write_addr = get_write_ptr(cb0_id);
        uint64_t read_addr = get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], tensor_address0);
        read_addr += shard_tile_id * tensor0_page_size;
        DeviceZoneScopedN("CCL WRD read tile ");
        noc_async_read(read_addr, l1_write_addr, num_tiles_to_read_this_core * tensor0_page_size);
        // noc_async_read_barrier();

        // cb_push_back(cb0_id, num_tiles_to_read_this_core);
        l1_write_addr += num_tiles_to_read_this_core * tensor0_page_size;
        tiles_read += num_tiles_to_read_this_core;
        shard_tile_id = 0;
        core_id++;
    }
    noc_async_read_barrier();
    cb_push_back(cb0_id, num_tiles_to_read);
    core_id = 0;
    DPRINT << "CCL WRD core_noc_x_to_write_to[core_id]: " << core_noc_x_to_write_to[core_id] << ENDL();
    DPRINT << "CCL WRD core_noc_y_to_write_to[core_id]: " << core_noc_y_to_write_to[core_id] << ENDL();

    uint32_t noc0_x_temp = static_cast<uint32_t>(out_ready_sem_noc0_x);
    uint32_t noc0_y_temp = static_cast<uint32_t>(out_ready_sem_noc0_y);

    DPRINT << "CCL WRD out_ready_sem_noc0_x: " << noc0_x_temp << ENDL();
    DPRINT << "CCL WRD out_ready_sem_noc0_y: " << noc0_y_temp << ENDL();

    uint64_t noc0_dest_noc_addr = get_noc_addr(
        core_noc_x_to_write_to[core_id], core_noc_y_to_write_to[core_id], intermediate_tensor_address0, 0 /*noc_id*/);
    uint32_t l1_read_addr = get_read_ptr(cb0_id);

    {
        DeviceZoneScopedN("CCL WRD write tile ");
        // comment out
        noc_async_write(
            l1_read_addr,
            noc0_dest_noc_addr + intermediate_first_core_tile_start_offset * tensor0_page_size,
            num_tiles_to_read * tensor0_page_size);
    }

    {
        DeviceZoneScopedN("CCL WRD sem incr ");
        // notify local receiver core
        uint64_t out_ready_sem_noc_addr =
            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
        noc_semaphore_inc(out_ready_sem_noc_addr, 1);
        noc_async_write_barrier();
        noc_async_atomic_barrier();
    }
}
