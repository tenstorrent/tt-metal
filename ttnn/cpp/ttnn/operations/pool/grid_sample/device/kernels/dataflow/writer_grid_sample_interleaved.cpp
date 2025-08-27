// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_to_write = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t ntiles_c = get_compile_time_arg_val(2);

    constexpr auto dst_args = TensorAccessorArgs<3>();

    const auto s0 = TensorAccessor(dst_args, dst_addr, output_stick_size);

    uint32_t end_stick_id = start_stick_id + num_sticks_to_write;

    // For grid sample: output is row major, each stick is written directly
    // We wait for ntiles_c pages to accumulate one full output stick

    DPRINT << "Writer: Writing " << (uint32_t)(end_stick_id - start_stick_id) << " sticks starting from stick id "
           << (uint32_t)start_stick_id << "\n";

    for (uint32_t stick_id = start_stick_id; stick_id < end_stick_id; stick_id++) {
        {
            {
                DeviceZoneScopedN("Waiting for output stick");

                cb_wait_front(cb_id_out0, ntiles_c);
            }
            // Wait for ntiles_c pages in output CB (one full stick)

            // Get base read address for this stick's data
            uint64_t base_l1_read_addr = get_read_ptr(cb_id_out0);

            // For row major grid sample output, we write one complete stick

            uint64_t dst_noc_addr = s0.get_noc_addr(stick_id);

            // Write the complete stick (ntiles_c * TILE_WIDTH elements)
            // The data is already laid out correctly in the CB pages
            {
                DeviceZoneScopedN("Write + barrier");
                // noc_async_write(base_l1_read_addr, dst_noc_addr, output_stick_size);

                // noc_async_write_barrier();
            }

            // Pop the ntiles_c pages we just consumed
            cb_pop_front(cb_id_out0, ntiles_c);
        }
    }
}
