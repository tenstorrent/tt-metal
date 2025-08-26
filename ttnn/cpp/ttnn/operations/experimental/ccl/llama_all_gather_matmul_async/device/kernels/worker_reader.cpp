// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include <cstdint>
#include <utility>
#include "debug/dprint.h"

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
    // DPRINT << "reader worker_reader args loaded" << ENDL();
    //  interleaved addrgen

    uint32_t tiles_read = 0;
    uint32_t shard_tile_id = first_core_tile_start_offset;
    uint32_t core_id = 0;
    cb_reserve_back(cb0_id, num_tiles_to_read);
    // DPRINT << "reader cb_reserve_back done" << ENDL();
    uint32_t l1_write_addr = get_write_ptr(cb0_id);
    while (tiles_read < num_tiles_to_read) {
        uint32_t num_tiles_to_read_this_core =
            std::min(num_tiles_per_core - shard_tile_id, num_tiles_to_read - tiles_read);
        // cb_reserve_back(cb0_id, num_tiles_to_read_this_core);
        // const uint32_t l1_write_addr = get_write_ptr(cb0_id);
        uint64_t read_addr = get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], tensor_address0);
        read_addr += shard_tile_id * tensor0_page_size;
        // DPRINT << "reader read_addr: " << ENDL();
        noc_async_read(read_addr, l1_write_addr, num_tiles_to_read_this_core * tensor0_page_size);
        // noc_async_read_barrier();
        // DPRINT << "reader noc_async_read done" << ENDL();
        // cb_push_back(cb0_id, num_tiles_to_read_this_core);
        l1_write_addr += num_tiles_to_read_this_core * tensor0_page_size;
        tiles_read += num_tiles_to_read_this_core;
        shard_tile_id = 0;
        core_id++;
    }
    noc_async_read_barrier();
    // DPRINT << "reader noc_async_read_barrier done" << ENDL();
    cb_push_back(cb0_id, num_tiles_to_read);
    // DPRINT << "reader cb_push_back done" << ENDL();
    core_id = 0;
    uint64_t noc0_dest_noc_addr = get_noc_addr(
        core_noc_x_to_write_to[core_id], core_noc_y_to_write_to[core_id], intermediate_tensor_address0, 0 /*noc_id*/);
    uint32_t l1_read_addr = get_read_ptr(cb0_id);
    noc_async_write(
        l1_read_addr,
        noc0_dest_noc_addr + intermediate_first_core_tile_start_offset * tensor0_page_size,
        num_tiles_to_read * tensor0_page_size);
    // DPRINT << "reader noc_async_write done" << ENDL();

    // notify local receiver core
    uint64_t out_ready_sem_noc_addr =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
    noc_semaphore_inc(out_ready_sem_noc_addr, 1);
    // DPRINT << "reader semaphore inc done" << ENDL();
    noc_async_write_barrier();
    // DPRINT << "reader noc_async_write_barrier done" << ENDL();
    noc_async_atomic_barrier();
    // DPRINT << "reader noc_async_atomic_barrier done" << ENDL();
}
