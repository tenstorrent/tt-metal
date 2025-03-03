// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "fabric/hw/inc/tt_fabric.h"
#include "fabric/hw/inc/tt_fabric_interface.h"
#include "fabric/hw/inc/tt_fabric_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"

// clang-format on

using namespace tt::tt_fabric;

volatile fabric_client_interface_t* client_interface;

uint64_t xy_local_addr;

void kernel_main() {
    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(1);
    constexpr uint32_t lower_pages = get_compile_time_arg_val(2);
    constexpr uint32_t higher_pages = get_compile_time_arg_val(3);

    uint32_t rt_args_idx = 0;
    uint32_t src_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_bytes = get_arg_val<uint32_t>(rt_args_idx++);

    // DPRINT << "lower_pages: " << (uint32_t)lower_pages << ENDL();
    // DPRINT << "higher_pages: " << (uint32_t)higher_pages << ENDL();
    // DPRINT << "src address: " << (uint32_t)src_addr << ENDL();
    // DPRINT << "num_bytes: " << (uint32_t)num_bytes << ENDL();
    const DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src0_is_dram> s = {
        .bank_base_address = src_addr, .page_size = num_bytes, .data_format = data_format};

    cb_reserve_back(cb_id_in0, higher_pages * lower_pages);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
    uint32_t original_addr = get_write_ptr(cb_id_in0);

    for (uint32_t i = 0; i < higher_pages * lower_pages; i++) {
        uint64_t src_noc_addr = get_noc_addr(i, s);
        // noc_async_read(src_noc_addr, l1_write_addr, num_bytes);
        noc_async_read_tile(i, s, l1_write_addr);
        l1_write_addr += num_bytes;
    }
    noc_async_read_barrier();
    /*
    auto* ptr_orig = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(original_addr);
    for (uint32_t ii1 = 0; ii1 < 1024; ii1 = ii1 + 1) {
        if (ii1 % 16 == 0) {
            DPRINT << "CHECK HERE ";
        }
        DPRINT << "value at i1 = " << (uint32_t)ii1 << " is: " << BF16((uint16_t)ptr_orig[ii1]) << ENDL();
    }
    */
    cb_push_back(cb_id_in0, higher_pages * lower_pages);
}
