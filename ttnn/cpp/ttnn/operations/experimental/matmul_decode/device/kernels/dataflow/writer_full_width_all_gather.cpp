// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_local_output.hpp"

void kernel_main() {
    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    all_gather_open_connections(fabric_connection);
    all_gather_local_output(fabric_connection);
}
