// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_local_output.hpp"

void kernel_main() {
    constexpr bool use_mux = get_named_compile_time_arg_val("ag_use_mux") != 0;
    if constexpr (use_mux) {
        uint32_t arg_idx =
            get_named_compile_time_arg_val("ag_rt_arg_base") + 4 + 2 * get_named_compile_time_arg_val("ag_num_shards");
        constexpr uint32_t mux_buffers = get_named_compile_time_arg_val("ag_mux_num_buffers");
        auto mux_connections = std::array{
            all_gather_parse_mux_connection<mux_buffers>(arg_idx),
            all_gather_parse_mux_connection<mux_buffers>(arg_idx)};
        all_gather_open_mux_connections(mux_connections.data());
        tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
        all_gather_local_output<mux_buffers>(fabric_connection, mux_connections.data());
    } else {
        tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
        all_gather_open_connections(fabric_connection);
        all_gather_local_output<2>(fabric_connection);
    }
}
