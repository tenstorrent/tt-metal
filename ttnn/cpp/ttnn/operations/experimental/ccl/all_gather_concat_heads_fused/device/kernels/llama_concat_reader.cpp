// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_constants.hpp>
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/fabric_connection_manager.hpp"
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

    size_t arg_idx = 0;

    const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t temp_cb_id = get_compile_time_arg_val(0);

    DPRINT << "temp_cb_id: " << (uint32_t)temp_cb_id << ENDL();
    DPRINT << "out_ready_sem_bank_addr: " << (uint32_t)out_ready_sem_bank_addr << ENDL();
    DPRINT << "out_ready_sem_wait_value: " << (uint32_t)out_ready_sem_wait_value << ENDL();
    DPRINT << "out_ready_sem_noc0_x: " << (uint32_t)out_ready_sem_noc0_x << ENDL();
    DPRINT << "out_ready_sem_noc0_y: " << (uint32_t)out_ready_sem_noc0_y << ENDL();
    uint64_t out_ready_sem_noc_addr =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
    auto temp_writer_addr = get_write_ptr(temp_cb_id);
    noc_async_read(out_ready_sem_noc_addr, temp_writer_addr, 2);
    volatile tt_l1_ptr uint16_t* tmp_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(temp_writer_addr);
    while (tmp_ptr[0] < out_ready_sem_wait_value) {
        noc_async_read(out_ready_sem_noc_addr, temp_writer_addr, 2);
        tmp_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(temp_writer_addr);
    }

    DPRINT << "AFTER WAIT VAL done, CAN READ SEMAPHORE VAL\n";
}
