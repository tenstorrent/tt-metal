// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp"

void kernel_main() {
    // Define all compile-time arguments at the beginning
    constexpr uint32_t fabric_receiver_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t accumulator_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_devices = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_core_width_output = get_compile_time_arg_val(3);
    constexpr uint32_t num_pages_per_packet = get_compile_time_arg_val(4);

    // Hardware startup stays with the kernel; the helper owns only the summation.
    compute_kernel_hw_startup(fabric_receiver_cb_id, fabric_receiver_cb_id, accumulator_cb_id);

    // Sum the per-device page blocks resident in the fabric-receiver CB into one output block.
    // The pre-migration pair loop accumulated with acc_to_dest from DST's zero start — sound per
    // the helper's DST-zero invariant (release ZEROACCs), and preserved bit-identically for even
    // device counts; odd counts (previously unsupported) copy_tile-seed with device 0's block.
    compute_kernel_lib::sum_blocks(
        fabric_receiver_cb_id, accumulator_cb_id, num_devices, num_pages_per_packet, /*pop_input=*/true);
}
