// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "full_width_rms_norm_transport.hpp"

void kernel_main() {
    uint32_t arg = 0;
    const bool is_hub = get_arg_val<uint32_t>(arg++) != 0;
    const uint32_t hub_x = get_arg_val<uint32_t>(arg++);
    const uint32_t hub_y = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_start_x = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_start_y = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_end_x = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_end_y = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_num_cores = get_arg_val<uint32_t>(arg++);  // includes the hub: the scale mcast loops back
    const uint32_t producer_index = get_arg_val<uint32_t>(arg++);
    run_full_width_rms_norm_transport(
        is_hub, hub_x, hub_y, mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, mcast_num_cores, producer_index);
}
