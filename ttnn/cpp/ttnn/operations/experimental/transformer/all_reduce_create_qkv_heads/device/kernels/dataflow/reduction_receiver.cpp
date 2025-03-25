// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "risc_attribs.h"
#include <tt-metalium/constants.hpp>
#include "debug/dprint.h"
using namespace tt::constants;

inline __attribute__((always_inline)) void fill_pad_cb_with_val(
    const uint32_t cb_id, const uint32_t num_bytes, const uint32_t val) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));

    for (uint32_t i = 0; i < num_bytes / 2; ++i) {
        ptr[i] = val;
    }
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    // reduction CT args
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t total_num_reduction_tiles = get_compile_time_arg_val(1);
    // QKV heads CT args
    constexpr uint32_t ELEMENT_SIZE = get_compile_time_arg_val(2);
    constexpr uint32_t SUBTILE_LINE_BYTES = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_q_out = get_compile_time_arg_val(4);
    constexpr uint32_t cb_id_k_out = get_compile_time_arg_val(5);
    constexpr uint32_t cb_id_v_out = get_compile_time_arg_val(6);
    constexpr uint32_t head_size = get_compile_time_arg_val(7);
    constexpr uint32_t num_q_heads = get_compile_time_arg_val(8);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(9);
    constexpr uint32_t head_size_num_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t PHASES_TO_READ =
        get_compile_time_arg_val(11);  // 0 to read all phases, 1 to read only first phase, 2 to read only second phase
    constexpr uint32_t in_num_cores = get_compile_time_arg_val(12);
    constexpr uint32_t q_num_cores = get_compile_time_arg_val(13);
    constexpr uint32_t k_num_cores = get_compile_time_arg_val(14);
    constexpr bool use_batch_offset = get_compile_time_arg_val(15) == 1;
    constexpr bool index_is_dram = get_compile_time_arg_val(16) == 1;
    constexpr uint32_t index_stick_size = get_compile_time_arg_val(17);
    constexpr uint32_t cb_batch_offset_id = get_compile_time_arg_val(18);
    constexpr uint32_t cb_id_reduction_out = get_compile_time_arg_val(19);
    constexpr uint32_t cb_id_zero = get_compile_time_arg_val(20);
    // Print all compile-time arguments
    DPRINT << "All compile-time arguments:" << ENDL();
    DPRINT << "cb_id = " << cb_id << ENDL();
    DPRINT << "total_num_reduction_tiles = " << total_num_reduction_tiles << ENDL();
    DPRINT << "ELEMENT_SIZE = " << ELEMENT_SIZE << ENDL();
    DPRINT << "SUBTILE_LINE_BYTES = " << SUBTILE_LINE_BYTES << ENDL();
    DPRINT << "cb_id_q_out = " << cb_id_q_out << ENDL();
    DPRINT << "cb_id_k_out = " << cb_id_k_out << ENDL();
    DPRINT << "cb_id_v_out = " << cb_id_v_out << ENDL();
    DPRINT << "head_size = " << head_size << ENDL();
    DPRINT << "num_q_heads = " << num_q_heads << ENDL();
    DPRINT << "num_kv_heads = " << num_kv_heads << ENDL();
    DPRINT << "head_size_num_tiles = " << head_size_num_tiles << ENDL();
    DPRINT << "PHASES_TO_READ = " << PHASES_TO_READ << ENDL();
    DPRINT << "in_num_cores = " << in_num_cores << ENDL();
    DPRINT << "q_num_cores = " << q_num_cores << ENDL();
    DPRINT << "k_num_cores = " << k_num_cores << ENDL();
    // DPRINT << "use_batch_offset = " << use_batch_offset << ENDL();
    // DPRINT << "index_is_dram = " << index_is_dram << ENDL();
    DPRINT << "index_stick_size = " << index_stick_size << ENDL();
    DPRINT << "cb_batch_offset_id = " << cb_batch_offset_id << ENDL();
    DPRINT << "cb_id_reduction_out = " << cb_id_reduction_out << ENDL();
    DPRINT << "cb_id_zero = " << cb_id_zero << ENDL();
    // runtime args
    size_t arg_idx = 0;
    // rt args for QV/K read and write kernels
    uint32_t q_start_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t k_start_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t v_start_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t batch_offset_tensor_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t index_in_cores = get_arg_val<uint32_t>(arg_idx++);
    uint32_t qv_k_process_flag = get_arg_val<uint32_t>(arg_idx++);
    uint32_t block_num_tiles = get_arg_val<uint32_t>(arg_idx++);

    // Print runtime arguments
    DPRINT << "Runtime arguments:" << ENDL();
    DPRINT << "q_start_addr = " << q_start_addr << ENDL();
    DPRINT << "k_start_addr = " << k_start_addr << ENDL();
    DPRINT << "v_start_addr = " << v_start_addr << ENDL();
    DPRINT << "batch_offset_tensor_addr = " << batch_offset_tensor_addr << ENDL();
    DPRINT << "index_in_cores = " << index_in_cores << ENDL();
    DPRINT << "qv_k_process_flag = " << qv_k_process_flag << ENDL();
    DPRINT << "block_num_tiles = " << block_num_tiles << ENDL();

    // tt_l1_ptr uint32_t* in0_mcast_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    // tt_l1_ptr uint32_t* in0_mcast_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx + in_num_cores));

    tt_l1_ptr uint32_t* q_in0_mcast_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    tt_l1_ptr uint32_t* q_in0_mcast_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx + q_num_cores));

    tt_l1_ptr uint32_t* k_in0_mcast_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx + 2 * q_num_cores));
    tt_l1_ptr uint32_t* k_in0_mcast_noc_y =
        (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx + 2 * q_num_cores + k_num_cores));

    // Print NOC addresses
    DPRINT << "q_in0_mcast_noc_x = " << (uint32_t)q_in0_mcast_noc_x << ENDL();
    DPRINT << "q_in0_mcast_noc_y = " << (uint32_t)q_in0_mcast_noc_y << ENDL();
    DPRINT << "k_in0_mcast_noc_x = " << (uint32_t)k_in0_mcast_noc_x << ENDL();
    DPRINT << "k_in0_mcast_noc_y = " << (uint32_t)k_in0_mcast_noc_y << ENDL();

    // rt args for reduction receiver kernel
    const uint32_t signal_semaphore_addr =
        get_semaphore(get_arg_val<uint32_t>(arg_idx + 2 * q_num_cores + 2 * k_num_cores));

    DPRINT << "signal_semaphore_addr = " << signal_semaphore_addr << ENDL();

    if constexpr (PHASES_TO_READ == 1) {  // only do the semaphore in reading kernel(NCRISC), as all reduce reduction
                                          // only has reading kernel
        DPRINT << "PHASES_TO_READ = " << PHASES_TO_READ << " which is NCRISC" << ENDL();
        volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);

        // 1. Wait for signal from All-Gather worker
        noc_semaphore_wait(signal_semaphore_addr_ptr, VALID);
        noc_semaphore_set(signal_semaphore_addr_ptr, 0);

        // 2. Signal compute kernel to start processing
        cb_push_back(cb_id, total_num_reduction_tiles);
        DPRINT << "total_num_reduction_tiles = " << total_num_reduction_tiles << " which is NCRISC" << ENDL();
    }
    // 3. QV/K read and write kernels start here:
    // 3.1 set up the device batch offset for each of the device.

    // fill the zero cb with 0
    fill_pad_cb_with_val(cb_id_zero, SUBTILE_LINE_BYTES, 0);

    uint32_t device_batch_offset = 0;

    if constexpr (use_batch_offset) {
        const InterleavedAddrGen<index_is_dram> addrg = {
            .bank_base_address = batch_offset_tensor_addr, .page_size = index_stick_size};
        cb_reserve_back(cb_batch_offset_id, 1);
        uint32_t index_cb_wr_ptr = get_write_ptr(cb_batch_offset_id);
        // Read the batch offset 1 page to read
        uint64_t batch_offset_index_noc_addr = get_noc_addr(0, addrg);
        noc_async_read(batch_offset_index_noc_addr, index_cb_wr_ptr, index_stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_batch_offset_id, 1);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_wr_ptr);
        // Always pick 1st value in tensor as batch offset
        device_batch_offset = index_ptr[0];
        DPRINT << "device_batch_offset = " << device_batch_offset << ENDL();
    }
    // device_batch_offset += index_in_cores;
    // uint32_t in_tile_offset_by_batch = device_batch_offset < 16
    //                                        ? device_batch_offset * SUBTILE_LINE_BYTES
    //                                        : (device_batch_offset - 16) * SUBTILE_LINE_BYTES + 512 * ELEMENT_SIZE;

    // 3.2 start to process the reading side of data(half of the data processed in NCRISC kernel and half on BRISC
    // kernel)
    DPRINT << "waiting for reduction out" << ENDL();
    cb_wait_front(cb_id_reduction_out, block_num_tiles);
    DPRINT << "reduction out done" << ENDL();

    constexpr uint32_t tile_size = head_size / head_size_num_tiles;
    constexpr uint32_t HALF_TILE_ELEMENTS = FACE_HEIGHT * TILE_WIDTH;
    constexpr uint32_t SUBTILE_ROWS = FACE_HEIGHT;

    // 3.2.1 read the QV/K data from the noc
    for (uint32_t i_output_core = 0; i_output_core < q_num_cores;
         ++i_output_core) {  // i_output_core is also the row number modulo 8 from input tile
        uint32_t device_batch_offset_per_output_core = device_batch_offset + i_output_core;
        uint32_t in_tile_offset_by_batch =
            device_batch_offset_per_output_core < 16
                ? device_batch_offset_per_output_core * SUBTILE_LINE_BYTES
                : (device_batch_offset_per_output_core - 16) * SUBTILE_LINE_BYTES + 512 * ELEMENT_SIZE;

        uint32_t read_addr = get_read_ptr(cb_id_reduction_out) + in_tile_offset_by_batch;
        uint64_t write_addr = 0;
        uint32_t head_index = 0, tile_index = 0, wptr_offset = 0;
        // i_output_core < SUBTILE_ROWS ? i_output_core * SUBTILE_LINE_BYTES
        //                  : (i_output_core - SUBTILE_ROWS) * SUBTILE_LINE_BYTES + HALF_TILE_ELEMENTS * ELEMENT_SIZE;
        if (index_in_cores < num_q_heads * head_size_num_tiles) {
            // read Q
            head_index = index_in_cores / head_size_num_tiles;  // head index of current tile(sits in current core)
            tile_index = index_in_cores % head_size_num_tiles;  // tile index of current tile(sits in current core)
            wptr_offset = head_index * SUBTILE_LINE_BYTES + tile_index * tile_size;
            write_addr =
                get_noc_addr(q_in0_mcast_noc_x[i_output_core], q_in0_mcast_noc_y[i_output_core], q_start_addr) +
                wptr_offset;
        } else if (index_in_cores < (num_q_heads + num_kv_heads) * head_size_num_tiles) {
            // read K
            head_index = 0;                                     // head index of current tile(sits in current core)
            tile_index = index_in_cores % head_size_num_tiles;  // tile index of current tile(sits in current core)
            wptr_offset = head_index * SUBTILE_LINE_BYTES + tile_index * tile_size;
            uint32_t tile_index_in_output_core = index_in_cores - num_q_heads * head_size_num_tiles;
            write_addr =
                get_noc_addr(k_in0_mcast_noc_x[i_output_core], k_in0_mcast_noc_y[i_output_core], k_start_addr) +
                wptr_offset;
        } else {
            // read V
            head_index = 0;                                     // head index of current tile(sits in current core)
            tile_index = index_in_cores % head_size_num_tiles;  // tile index of current tile(sits in current core)
            wptr_offset = head_index * SUBTILE_LINE_BYTES + tile_index * tile_size;
            uint32_t tile_index_in_output_core = index_in_cores - (num_q_heads + num_kv_heads) * head_size_num_tiles;
            write_addr =
                get_noc_addr(q_in0_mcast_noc_x[i_output_core], q_in0_mcast_noc_y[i_output_core], v_start_addr) +
                wptr_offset;
        }

        if constexpr (PHASES_TO_READ == 1) {  // reader kernel (NCRISC)
            DPRINT << "NCRISC WRITE" << ENDL();
            noc_async_write(read_addr, write_addr, SUBTILE_LINE_BYTES);
            // noc_async_write(get_read_ptr(cb_id_zero), write_addr, SUBTILE_LINE_BYTES);
        }
        if constexpr (PHASES_TO_READ == 2) {  // writer kernel(BRISC)
            DPRINT << "BRISC WRITE" << ENDL();
            noc_async_write(
                read_addr + FACE_HW * ELEMENT_SIZE, write_addr + FACE_HW * ELEMENT_SIZE, SUBTILE_LINE_BYTES);
            // noc_async_write(get_read_ptr(cb_id_zero), write_addr + FACE_HW * ELEMENT_SIZE, SUBTILE_LINE_BYTES);
        }
        noc_async_read_barrier();
    }
    noc_async_read_barrier();
}
