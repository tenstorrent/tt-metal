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
#include <array>

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
    constexpr uint32_t batch_size = get_compile_time_arg_val(11);
    constexpr uint32_t batch_start_1 = get_compile_time_arg_val(12);
    constexpr uint32_t batch_end_1 = get_compile_time_arg_val(13);
    constexpr uint32_t batch_start_2 = get_compile_time_arg_val(14);
    constexpr uint32_t batch_end_2 = get_compile_time_arg_val(15);
    constexpr uint32_t start_local = get_compile_time_arg_val(16);

    DPRINT << "batch_size: " << (uint32_t)batch_size << ENDL();
    DPRINT << "batch_start_1: " << (uint32_t)batch_start_1 << ENDL();
    DPRINT << "batch_end_1: " << (uint32_t)batch_end_1 << ENDL();
    DPRINT << "batch_start_2: " << (uint32_t)batch_start_2 << ENDL();
    DPRINT << "batch_end_2: " << (uint32_t)batch_end_2 << ENDL();
    DPRINT << "start_local: " << (uint32_t)start_local << ENDL();

    uint32_t arg_idx = 3 + 2 * in_num_cores;
    uint32_t tensor_address0 = get_arg_val<uint32_t>(arg_idx);

    std::array<uint32_t, 8> core_noc_x = {19, 20, 21, 19, 20, 21, 19, 20};
    std::array<uint32_t, 8> core_noc_y = {18, 18, 18, 19, 19, 19, 20, 20};

    DPRINT << "this core runs concat\n";
    DPRINT << "temp_cb_id: " << (uint32_t)temp_cb_id << ENDL();

    // uint64_t out_ready_sem_noc_addr =
    //     safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);

    DPRINT << "RT ARGS HERE\n";
    DPRINT << "in_tile_offset_by_head: " << in_tile_offset_by_head << ENDL();
    DPRINT << "q_start_addr: " << q_start_addr << ENDL();
    DPRINT << "signal semaphore addr: " << (uint32_t)signal_semaphore_addr << ENDL();
    // DPRINT << "out_ready_sem_bank_addr: " << (uint32_t)out_ready_sem_bank_addr << ENDL();
    // DPRINT << "out_ready_sem_wait_value: " << (uint32_t)out_ready_sem_wait_value << ENDL();
    // DPRINT << "out_ready_sem_noc0_x: " << (uint32_t)out_ready_sem_noc0_x << ENDL();
    // DPRINT << "out_ready_sem_noc0_y: " << (uint32_t)out_ready_sem_noc0_y << ENDL();
    for (uint32_t i = 0; i < 8; i++) {
        DPRINT << "core_noc_x[" << i << "]: " << (uint32_t)core_noc_x[i] << "\n";
        DPRINT << "core_noc_y[" << i << "]: " << (uint32_t)core_noc_y[i] << "\n";
    }
    DPRINT << "tensor_address0: " << (uint32_t)(tensor_address0) << ENDL();
    DPRINT << "END OF RT ARGS\n";

    auto batch_loop = [&](uint32_t head_size_num_tiles,
                          uint32_t q_start_addr,
                          uint32_t tensor_address0,
                          uint32_t face_h,
                          uint32_t SUBTILE_LINE_BYTES,
                          uint32_t face_hw,
                          uint32_t ELEMENT_SIZE,
                          const uint32_t cb_write_ptr_base,
                          uint64_t qkv_read_addr,
                          uint32_t tile_size,
                          uint32_t num_tiles_read_cur_core,
                          uint32_t cur_core_idx,
                          uint32_t num_tiles_per_core_concat,
                          uint32_t start,
                          uint32_t end,
                          uint32_t local_count,
                          std::array<uint32_t, 8> core_noc_x,
                          std::array<uint32_t, 8> core_noc_y,
                          bool nlp_local) {
        tt_l1_ptr uint32_t* in0_mcast_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(3));
        tt_l1_ptr uint32_t* in0_mcast_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(3 + in_num_cores));

        DPRINT << "start: " << start << ENDL();
        DPRINT << "end:  " << end << ENDL();

        for (uint32_t q = start; q < end; ++q) {
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
                        qkv_read_addr + face_hw * ELEMENT_SIZE,
                        q_write_addr + face_hw * ELEMENT_SIZE,
                        SUBTILE_LINE_BYTES);
                }
                qkv_read_addr += tile_size;
                q_write_addr += tile_size;
                num_tiles_read_cur_core++;

                if (num_tiles_read_cur_core == num_tiles_per_core_concat) {
                    cur_core_idx++;
                    local_count++;
                    qkv_read_addr =
                        get_noc_addr(in0_mcast_noc_x[cur_core_idx], in0_mcast_noc_y[cur_core_idx], q_start_addr) +
                        in_tile_offset_by_head;
                    if (nlp_local) {
                        qkv_read_addr =
                            get_noc_addr(core_noc_x[local_count], core_noc_y[local_count], tensor_address0) +
                            in_tile_offset_by_head;
                    }
                    num_tiles_read_cur_core = 0;
                }
            }
        }
    };

    auto nlp_concat = [&](uint32_t head_size_num_tiles,
                          uint32_t batch,
                          uint32_t q_start_addr,
                          uint32_t tensor_address0,
                          uint32_t head_size,
                          uint32_t cb_id_q_out,
                          uint32_t face_h,
                          uint32_t SUBTILE_LINE_BYTES,
                          uint32_t face_hw,
                          uint32_t ELEMENT_SIZE,
                          bool nlp_local,
                          uint32_t start_local,
                          std::array<uint32_t, 8> core_noc_x,
                          std::array<uint32_t, 8> core_noc_y) {
        tt_l1_ptr uint32_t* in0_mcast_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(3));
        tt_l1_ptr uint32_t* in0_mcast_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(3 + in_num_cores));
        // Q
        uint32_t cur_core_idx = batch_start_1;

        uint32_t total_input_cores = in_num_cores;
        uint32_t num_tiles_per_core_concat = (head_size_num_tiles * batch) / total_input_cores;

        uint64_t qkv_read_addr =
            get_noc_addr(in0_mcast_noc_x[cur_core_idx], in0_mcast_noc_y[cur_core_idx], q_start_addr) +
            in_tile_offset_by_head;
        uint32_t local_count = 0;
        if (nlp_local) {
            qkv_read_addr = get_noc_addr(core_noc_x[local_count], core_noc_y[local_count], tensor_address0) +
                            in_tile_offset_by_head;
            DPRINT << "reading from: " << core_noc_x[local_count] << " " << core_noc_y[local_count] << ENDL();
        } else {
            DPRINT << "reading from: " << in0_mcast_noc_x[cur_core_idx] << " " << in0_mcast_noc_y[cur_core_idx]
                   << ENDL();
        }
        uint32_t num_tiles_read_cur_core = 0;
        uint32_t q_write_addr = 0;
        uint32_t tile_size = head_size / head_size_num_tiles;
        const uint32_t cb_write_ptr_base = get_write_ptr(cb_id_q_out);

        uint32_t start = nlp_local ? start_local : batch_start_1;
        uint32_t end = nlp_local ? start_local + 8 : batch_end_1;
        uint32_t idx_end = nlp_local ? 1 : batch_size;

        for (uint32_t batch_range = 0; batch_range < idx_end; batch_range++) {
            batch_loop(
                head_size_num_tiles,
                q_start_addr,
                tensor_address0,
                face_h,
                SUBTILE_LINE_BYTES,
                face_hw,
                ELEMENT_SIZE,
                cb_write_ptr_base,
                qkv_read_addr,
                tile_size,
                num_tiles_read_cur_core,
                cur_core_idx,
                num_tiles_per_core_concat,
                start,
                end,
                local_count,
                core_noc_x,
                core_noc_y,
                nlp_local);
            start = batch_start_2;
            end = batch_end_2;
            cur_core_idx = batch_start_2;
            qkv_read_addr = get_noc_addr(in0_mcast_noc_x[cur_core_idx], in0_mcast_noc_y[cur_core_idx], q_start_addr) +
                            in_tile_offset_by_head;
            DPRINT << "batch2 reading from core: " << (uint32_t)(in0_mcast_noc_x[cur_core_idx]) << " and "
                   << (uint32_t)(in0_mcast_noc_y[cur_core_idx]) << ENDL();

            num_tiles_read_cur_core = 0;
        }

        noc_async_read_barrier();
    };

    nlp_concat(
        head_size_num_tiles,
        batch,
        q_start_addr,
        tensor_address0,
        head_size,
        cb_id_q_out,
        face_h,
        SUBTILE_LINE_BYTES,
        face_hw,
        ELEMENT_SIZE,
        1,
        start_local,
        core_noc_x,
        core_noc_y);
    DPRINT << "DONE LOCAL NLP\n";

    // 1. Wait for signal from All-Gather worker
    noc_semaphore_wait(signal_semaphore_addr_ptr, VALID);
    noc_semaphore_set(signal_semaphore_addr_ptr, 0);

    nlp_concat(
        head_size_num_tiles,
        batch,
        q_start_addr,
        tensor_address0,
        head_size,
        cb_id_q_out,
        face_h,
        SUBTILE_LINE_BYTES,
        face_hw,
        ELEMENT_SIZE,
        0,
        start_local,
        core_noc_x,
        core_noc_y);

    DPRINT << "DONE\n";
}
