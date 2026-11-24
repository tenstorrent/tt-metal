// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// parallel_packet_builders (perf experiment): ids shared by the helper / port kernels.
//
// H helper cores per (direction, link) port build COMPLETE bulk packets (meta + routed header + payload) and write them
// into the port core's packet ring; the port BRISC (the only owner of the router connection: one worker channel per
// router in 2D fabric) only forwards them. Packet i of the port's walk is built by helper i % H as its j = i / H-th
// packet, into port ring slot (j % D) * H + h. Port slot layout: [meta 16 B: payload bytes,
// extra_inc][header][payload].
#pragma once

#include <cstdint>

namespace ppb {
// program semaphores (re-initialized every launch; every cross-core write of a call is waited for within the call).
// Helper-side and port-side ids overlap on purpose (declared on disjoint cores; a semaphore id is the same L1 offset
// on every core): the port keeps 0..7 (the op's) + 8..8+H-1 + the fabric connection's own.
constexpr uint32_t SEM_MIRROR = 8;  // helper: last arrival count the port forwarded (written by the port NCRISC)
constexpr uint32_t SEM_CREDIT = 9;  // helper: own packets the port has sent and flushed (port BRISC)
constexpr uint32_t SEM_OWN = 10;    // helper: 1 = staged own prefix ready, 2 = own pages written (port NCRISC)
constexpr uint32_t SEM_READY0 = 8;  // port: + h = packets helper h has written into the ring (value j + 1)
constexpr uint32_t MAX_H = 6;
constexpr uint32_t META_BYTES = 16;
}  // namespace ppb
