// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_constants.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_ccl_common.hpp"
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

    size_t arg_idx = 1;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    uint32_t out_ready_sem_bank_addr_concat = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_core_tile_start_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;

    // print every compile and runtime arg in uint32_t
    DPRINT << "ct args: \n";
    DPRINT << "my_chip_id: " << (uint32_t)my_chip_id << "\n";
    DPRINT << "cb0_id: " << (uint32_t)cb0_id << "\n";
    DPRINT << "tensor0_page_size: " << (uint32_t)tensor0_page_size << "\n";

    DPRINT << "rt args: \n";
    DPRINT << "tensor_address0: " << (uint32_t)tensor_address0 << "\n";
    DPRINT << "num_tiles_per_core: " << (uint32_t)num_tiles_per_core << "\n";
    DPRINT << "num_tiles_to_read: " << (uint32_t)num_tiles_to_read << "\n";
    DPRINT << "first_core_tile_start_offset: " << (uint32_t)first_core_tile_start_offset << "\n";
    DPRINT << "num_cores: " << (uint32_t)num_cores << "\n";
    for (uint32_t i = 0; i < num_cores; i++) {
        DPRINT << "core_noc_x[" << i << "]: " << (uint32_t)core_noc_x[i] << "\n";
        DPRINT << "core_noc_y[" << i << "]: " << (uint32_t)core_noc_y[i] << "\n";
    }

    uint32_t concat_arg_start = get_arg_val<uint32_t>(0);
    uint32_t in_tile_offset_by_head = get_arg_val<uint32_t>(concat_arg_start);
    uint32_t q_start_addr = get_arg_val<uint32_t>(concat_arg_start + 1);

    constexpr uint32_t ELEMENT_SIZE = get_compile_time_arg_val(3);
    constexpr uint32_t SUBTILE_LINE_BYTES = get_compile_time_arg_val(4);
    constexpr uint32_t cb_id_q_out = get_compile_time_arg_val(5);
    constexpr uint32_t head_size = get_compile_time_arg_val(6);
    constexpr uint32_t batch = get_compile_time_arg_val(7);
    constexpr uint32_t head_size_num_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t PHASES_TO_READ =
        get_compile_time_arg_val(9);  // 0 to read all phases, 1 to read only first phase, 2 to read only second phase

    constexpr uint32_t in_num_cores = get_compile_time_arg_val(10);
    constexpr uint32_t face_h = get_compile_time_arg_val(11);
    constexpr uint32_t face_hw = get_compile_time_arg_val(12);

    constexpr uint32_t temp_cb_id = get_compile_time_arg_val(13);

    uint32_t arg_sem_idx = 2 + 2 * in_num_cores;
    DPRINT << "out_ready_sem_bank_addr_concat is at arg index: " << (uint32_t)(concat_arg_start + arg_sem_idx);
    // uint32_t out_ready_sem_bank_addr_concat = get_arg_val<uint32_t>(concat_arg_start + arg_sem_idx);
    uint32_t out_ready_sem_wait_value_concat = get_arg_val<uint32_t>(concat_arg_start + arg_sem_idx + 1);
    uint32_t out_ready_sem_noc0_x_concat = get_arg_val<uint32_t>(concat_arg_start + arg_sem_idx + 2);
    uint32_t out_ready_sem_noc0_y_concat = get_arg_val<uint32_t>(concat_arg_start + arg_sem_idx + 3);
    uint32_t is_drain_core = get_arg_val<uint32_t>(concat_arg_start + arg_sem_idx + 4);
    const uint32_t signal_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(concat_arg_start + arg_sem_idx + 5));

    DPRINT << "signal semaphore addr: " << (uint32_t)signal_semaphore_addr << ENDL();

    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);

    DPRINT << "temp_cb_id: " << (uint32_t)temp_cb_id << ENDL();
    DPRINT << "out_ready_sem_bank_addr_concat: " << (uint32_t)out_ready_sem_bank_addr_concat << ENDL();
    DPRINT << "out_ready_sem_wait_value: " << (uint32_t)out_ready_sem_wait_value_concat << ENDL();
    DPRINT << "out_ready_sem_noc0_x: " << (uint32_t)out_ready_sem_noc0_x_concat << ENDL();
    DPRINT << "out_ready_sem_noc0_y: " << (uint32_t)out_ready_sem_noc0_y_concat << ENDL();
    DPRINT << "concat arg start: " << (uint32_t)concat_arg_start << ENDL();

    DPRINT << "tensor -> CB: " << (uint32_t)cb0_id << "\n";

    uint32_t tiles_read = 0;
    uint32_t shard_tile_id = first_core_tile_start_offset;
    uint32_t core_id = 0;
    while (tiles_read < num_tiles_to_read) {
        DPRINT << "tiles_read: " << tiles_read << "\n";
        uint32_t num_tiles_to_read_this_core =
            std::min(num_tiles_per_core - shard_tile_id, num_tiles_to_read - tiles_read);
        cb_reserve_back(cb0_id, num_tiles_to_read_this_core);
        DPRINT << "num_tiles_to_read_this_core: " << num_tiles_to_read_this_core << ENDL();
        const uint32_t l1_write_addr = get_write_ptr(cb0_id);
        uint64_t read_addr = get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], tensor_address0);
        read_addr += shard_tile_id * tensor0_page_size;

        noc_async_read(read_addr, l1_write_addr, num_tiles_to_read_this_core * tensor0_page_size);
        noc_async_read_barrier();

        cb_push_back(cb0_id, num_tiles_to_read_this_core);
        tiles_read += num_tiles_to_read_this_core;
        shard_tile_id = 0;
        core_id++;
    }

    DPRINT << "DONE ALL GATHER READ\n";

    uint64_t out_ready_sem_noc_addr_concat =
        safe_get_noc_addr(out_ready_sem_noc0_x_concat, out_ready_sem_noc0_y_concat, out_ready_sem_bank_addr_concat);

    DPRINT << "is drain core: " << (uint32_t)is_drain_core << ENDL();
    if (is_drain_core == 1) {
        while (*reinterpret_cast<volatile uint32_t*>(out_ready_sem_bank_addr_concat) !=
               out_ready_sem_wait_value_concat) {
            DPRINT << "waitval done\n";
        }
    } else {
        noc_semaphore_wait(signal_semaphore_addr_ptr, VALID);
        // noc_semaphore_set(signal_semaphore_addr_ptr, 0);
    }

    tt_l1_ptr uint32_t* in0_mcast_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(2 + concat_arg_start));
    tt_l1_ptr uint32_t* in0_mcast_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(2 + in_num_cores + concat_arg_start));

    // Q
    uint32_t cur_core_idx = 0;
    uint32_t total_input_cores = in_num_cores;
    uint32_t num_tiles_per_core_concat = (head_size_num_tiles * batch) / total_input_cores;

    uint64_t qkv_read_addr = get_noc_addr(in0_mcast_noc_x[cur_core_idx], in0_mcast_noc_y[cur_core_idx], q_start_addr) +
                             in_tile_offset_by_head;

    uint32_t num_tiles_read_cur_core = 0;
    uint32_t q_write_addr = 0;
    uint32_t tile_size = head_size / head_size_num_tiles;
    const uint32_t cb_write_ptr_base = get_write_ptr(cb_id_q_out);

    for (uint32_t q = 0; q < batch; ++q) {
        uint32_t wptr_offset = q < face_h ? q * SUBTILE_LINE_BYTES : (q + face_h) * SUBTILE_LINE_BYTES;
        uint32_t q_write_addr = cb_write_ptr_base + wptr_offset;
        for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
            // Read first phase
            if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 1) {
                noc_async_read(qkv_read_addr, q_write_addr, SUBTILE_LINE_BYTES);
            }
            // Read second phase
            if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 2) {
                noc_async_read(
                    qkv_read_addr + face_hw * ELEMENT_SIZE, q_write_addr + face_hw * ELEMENT_SIZE, SUBTILE_LINE_BYTES);
            }

            qkv_read_addr += tile_size;
            q_write_addr += tile_size;
            num_tiles_read_cur_core++;

            if (num_tiles_read_cur_core == num_tiles_per_core_concat) {
                cur_core_idx++;
                qkv_read_addr =
                    get_noc_addr(in0_mcast_noc_x[cur_core_idx], in0_mcast_noc_y[cur_core_idx], q_start_addr) +
                    in_tile_offset_by_head;
                num_tiles_read_cur_core = 0;
            }
        }
    }

    noc_async_read_barrier();

    DPRINT << "DONE\n";
}
