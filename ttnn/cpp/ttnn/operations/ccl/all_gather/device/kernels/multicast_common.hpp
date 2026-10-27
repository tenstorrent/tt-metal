// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

#ifdef FABRIC_2D
#include "tt_metal/fabric/hw/inc/mesh/api.h"
namespace fabric_api = tt::tt_fabric::mesh::experimental;
using FabricRange = tt::tt_fabric::mesh::experimental::MeshMcastRange;
inline FabricRange make_fabric_range(uint8_t e, uint8_t w, uint8_t n, uint8_t s) { return FabricRange{e, w, n, s}; }
#else
#include "tt_metal/fabric/hw/inc/linear/api.h"
namespace fabric_api = tt::tt_fabric::linear::experimental;
using FabricRange = uint8_t;  // under 1D each connection carries a single hop count
// 1D has a single active direction, so exactly one slot is nonzero
inline FabricRange make_fabric_range(uint8_t e, uint8_t w, uint8_t n, uint8_t s) { return e + w + n + s; }
#endif
