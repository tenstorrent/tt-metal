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

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t in_tile_offset_by_head = get_arg_val<uint32_t>(0);
    uint32_t q_start_addr = get_arg_val<uint32_t>(1);
    const uint32_t signal_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(2));

    DPRINT << "signal semaphore addr: " << (uint32_t)signal_semaphore_addr << ENDL();

    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);

    constexpr uint32_t ELEMENT_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t SUBTILE_LINE_BYTES = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_q_out = get_compile_time_arg_val(2);
    constexpr uint32_t head_size = get_compile_time_arg_val(3);
    constexpr uint32_t batch = get_compile_time_arg_val(4);
    constexpr uint32_t head_size_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t PHASES_TO_READ =
        get_compile_time_arg_val(6);  // 0 to read all phases, 1 to read only first phase, 2 to read only second phase

    constexpr uint32_t in_num_cores = get_compile_time_arg_val(7);
    constexpr uint32_t face_h = get_compile_time_arg_val(8);
    constexpr uint32_t face_hw = get_compile_time_arg_val(9);

    constexpr uint32_t temp_cb_id = get_compile_time_arg_val(10);

    uint32_t sem_arg_start = 3 + 2 * in_num_cores;
    uint32_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(sem_arg_start);
    uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(sem_arg_start + 1);
    uint32_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(sem_arg_start + 2);
    uint32_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(sem_arg_start + 3);

    DPRINT << "this core runs concat\n";
    DPRINT << "temp_cb_id: " << (uint32_t)temp_cb_id << ENDL();
    DPRINT << "out_ready_sem_bank_addr: " << (uint32_t)out_ready_sem_bank_addr << ENDL();
    DPRINT << "out_ready_sem_wait_value: " << (uint32_t)out_ready_sem_wait_value << ENDL();
    DPRINT << "out_ready_sem_noc0_x: " << (uint32_t)out_ready_sem_noc0_x << ENDL();
    DPRINT << "out_ready_sem_noc0_y: " << (uint32_t)out_ready_sem_noc0_y << ENDL();
    uint64_t out_ready_sem_noc_addr =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);

    // 1. Wait for signal from All-Gather worker
    noc_semaphore_wait(signal_semaphore_addr_ptr, VALID);
    noc_semaphore_set(signal_semaphore_addr_ptr, 0);

    tt_l1_ptr uint32_t* in0_mcast_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(3));
    tt_l1_ptr uint32_t* in0_mcast_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(3 + in_num_cores));

    // Q
    uint32_t cur_core_idx = 0;
    uint32_t total_input_cores = in_num_cores;
    uint32_t num_tiles_per_core = (head_size_num_tiles * batch) / total_input_cores;

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

            if (num_tiles_read_cur_core == num_tiles_per_core) {
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
