// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Include first, before any fabric header: on Fabric2D / TorusXY the fabric headers (packet header type,
// RoutingPlaneConnectionManager slots, mesh API) depend on FABRIC_2D. The JIT build sets ROUTING_MODE for
// fabric kernels; this is the condition under which fabric_edm_packet_header.hpp picks HybridMeshPacketHeader.
// The host may also pass FABRIC_2D explicitly.
#pragma once
#if !defined(FABRIC_2D) && defined(ROUTING_MODE)
#include "tt_metal/fabric/hw/inc/fabric_routing_mode.h"
#if !(((ROUTING_MODE & (ROUTING_MODE_1D | ROUTING_MODE_LINE)) != 0) ||  \
      ((ROUTING_MODE & (ROUTING_MODE_1D | ROUTING_MODE_RING)) != 0)) && \
    (((ROUTING_MODE & (ROUTING_MODE_2D | ROUTING_MODE_MESH)) != 0) ||   \
     ((ROUTING_MODE & (ROUTING_MODE_2D | ROUTING_MODE_TORUS)) != 0))
#define FABRIC_2D 1
#endif
#endif
