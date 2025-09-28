// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"

namespace tt::point_to_point::common {

inline auto& connection_direction_collection(const bool dst_is_forward, FabricConnectionManager& fabric_connection) {
    if (dst_is_forward) {
        return fabric_connection.get_forward_connection();
    } else {
        return fabric_connection.get_backward_connection();
    }
}
}  // namespace tt::point_to_point::common
