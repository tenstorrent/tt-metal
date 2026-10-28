// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "internal/risc_attribs.h"
#include "hostdevcommon/fabric_common.h"

// This chip's coordinate along mesh axis `axis` (0: row, 1: column) of a mesh `mesh_cols` chips wide, from the
// chip id fabric's control plane writes into every Tensix core's routing info. Valid only in a program the host
// built after mesh_coord_from_fabric_available (../mesh_coord_from_fabric.hpp) held: that is what makes the chip
// id this chip's row-major index over the mesh.
inline uint32_t mesh_coord_from_fabric(uint32_t axis, uint32_t mesh_cols) {
    const auto* routing_info =
        reinterpret_cast<volatile tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const uint32_t chip_id = routing_info->my_device_id;
    return axis == 0 ? chip_id / mesh_cols : chip_id % mesh_cols;
}
