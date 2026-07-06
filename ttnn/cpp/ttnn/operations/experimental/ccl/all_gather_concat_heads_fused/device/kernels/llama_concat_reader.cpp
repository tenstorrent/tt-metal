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
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>
#include <array>

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t cb_id_q_out = get_compile_time_arg_val(0);
constexpr uint32_t ROWS_TO_READ = get_compile_time_arg_val(1);

constexpr uint32_t in_num_cores = get_compile_time_arg_val(2);

constexpr uint32_t batch_size = get_compile_time_arg_val(3);
constexpr uint32_t batch_start_1 = get_compile_time_arg_val(4);
constexpr uint32_t batch_end_1 = get_compile_time_arg_val(5);
constexpr uint32_t batch_start_2 = get_compile_time_arg_val(6);
constexpr uint32_t batch_end_2 = get_compile_time_arg_val(7);
constexpr uint32_t start_local = get_compile_time_arg_val(8);
constexpr uint32_t input_row_size = get_compile_time_arg_val(9);
constexpr uint32_t output_row_size = get_compile_time_arg_val(10);
void batch_loop(
    uint32_t q_start_addr,
    uint32_t tensor_address0,
    const uint32_t cb_write_ptr_base,
    const Noc& noc_obj,
    uint32_t qkv_noc_x,
    uint32_t qkv_noc_y,
    uint32_t qkv_addr,
    uint32_t cur_core_idx,
    uint32_t start,
    uint32_t end,
    uint32_t local_count,
    std::array<uint32_t, 8> core_noc_x,
    std::array<uint32_t, 8> core_noc_y,
    tt_l1_ptr uint32_t* in0_mcast_noc_x,
    tt_l1_ptr uint32_t* in0_mcast_noc_y,
    bool nlp_local,
    uint32_t second_half_core,
    uint32_t start_row) {
    const uint32_t read_offset = output_row_size * second_half_core + start_row * input_row_size;
    uint32_t q_write_addr = cb_write_ptr_base + start * output_row_size;
    for (uint32_t q = start; q < end; ++q) {
        noc_obj.async_read(
            UnicastEndpoint{},
            CoreLocalMem<uint8_t>(q_write_addr),
            output_row_size,
            {.noc_x = qkv_noc_x, .noc_y = qkv_noc_y, .addr = qkv_addr},
            {});
        q_write_addr += output_row_size;
        cur_core_idx++;
        local_count++;
        qkv_noc_x = in0_mcast_noc_x[cur_core_idx];
        qkv_noc_y = in0_mcast_noc_y[cur_core_idx];
        qkv_addr = q_start_addr + read_offset;
        if (nlp_local) {
            qkv_noc_x = core_noc_x[local_count];
            qkv_noc_y = core_noc_y[local_count];
            qkv_addr = tensor_address0 + read_offset;
        }
    }
};

void nlp_concat(
    const Noc& noc_obj,
    CircularBuffer& cb_q_out,
    uint32_t q_start_addr,
    uint32_t tensor_address0,
    bool nlp_local,
    uint32_t start_local,
    std::array<uint32_t, 8> core_noc_x,
    std::array<uint32_t, 8> core_noc_y,
    tt_l1_ptr uint32_t* in0_mcast_noc_x,
    tt_l1_ptr uint32_t* in0_mcast_noc_y,
    uint32_t second_half_core,
    uint32_t start_row) {
    uint32_t start = nlp_local ? start_local : batch_start_1;
    uint32_t end = nlp_local ? start_local + 8 : batch_end_1;
    uint32_t idx_end = nlp_local ? 1 : batch_size;
    uint32_t local_count = 0;

    uint32_t cur_core_idx = batch_start_1;

    const uint32_t read_offset = output_row_size * second_half_core + start_row * input_row_size;
    uint32_t qkv_noc_x = in0_mcast_noc_x[cur_core_idx];
    uint32_t qkv_noc_y = in0_mcast_noc_y[cur_core_idx];
    uint32_t qkv_addr = q_start_addr + read_offset;

    if (nlp_local) {
        qkv_noc_x = core_noc_x[local_count];
        qkv_noc_y = core_noc_y[local_count];
        qkv_addr = tensor_address0 + read_offset;
    }
    uint32_t q_write_addr = 0;
    const uint32_t cb_write_ptr_base = cb_q_out.get_write_ptr();

    for (uint32_t batch_range = 0; batch_range < idx_end; batch_range++) {
        batch_loop(
            q_start_addr,
            tensor_address0,
            cb_write_ptr_base,
            noc_obj,
            qkv_noc_x,
            qkv_noc_y,
            qkv_addr,
            cur_core_idx,
            start,
            end,
            local_count,
            core_noc_x,
            core_noc_y,
            in0_mcast_noc_x,
            in0_mcast_noc_y,
            nlp_local,
            second_half_core,
            start_row);
        start = batch_start_2;
        end = batch_end_2;
        cur_core_idx = batch_start_2;
        qkv_noc_x = in0_mcast_noc_x[cur_core_idx];
        qkv_noc_y = in0_mcast_noc_y[cur_core_idx];
        qkv_addr = q_start_addr + read_offset;
    }

    noc_obj.async_read_barrier();
};

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    uint32_t q_start_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tensor_address0 = get_arg_val<uint32_t>(arg_idx++);
    Semaphore<> signal_sem(get_arg_val<uint32_t>(arg_idx++));
    Semaphore<> signal_sem2(get_arg_val<uint32_t>(arg_idx++));

    tt_l1_ptr uint32_t* in0_mcast_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += in_num_cores;
    tt_l1_ptr uint32_t* in0_mcast_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += in_num_cores;
    uint32_t second_half_core = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_row = get_arg_val<uint32_t>(arg_idx++);

    std::array<uint32_t, 8> core_noc_x = {19, 20, 21, 19, 20, 21, 19, 20};
    std::array<uint32_t, 8> core_noc_y = {18, 18, 18, 19, 19, 19, 20, 20};

    Noc noc_obj;
    CircularBuffer cb_q_out(cb_id_q_out);

    nlp_concat(
        noc_obj,
        cb_q_out,
        q_start_addr,
        tensor_address0,
        1,
        start_local,
        core_noc_x,
        core_noc_y,
        in0_mcast_noc_x,
        in0_mcast_noc_y,
        second_half_core,
        start_row);
    if (ROWS_TO_READ == 1) {
        signal_sem.wait(1);
        signal_sem.set(0);
    } else if (ROWS_TO_READ == 2) {
        signal_sem2.wait(1);
        signal_sem2.set(0);
    }

    nlp_concat(
        noc_obj,
        cb_q_out,
        q_start_addr,
        tensor_address0,
        0,
        start_local,
        core_noc_x,
        core_noc_y,
        in0_mcast_noc_x,
        in0_mcast_noc_y,
        second_half_core,
        start_row);
    cb_q_out.push_back(2);
}
