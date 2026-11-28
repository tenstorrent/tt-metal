// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Stage zones of the g6_clone kernels, compiled in only for the zone build: the host defines
// G6_CLONE_ZONES when the caller asks for it (g6_clone_program_descriptor.ZONE_BUILD_ENV). A plain
// build, profiled or not, carries no user zone, so its binaries and kernel times are the op's own
// (BB-21); in a profiled build each zone run costs about 50 cycles, a sixth of a one-page copy.
// Every zone stays at its stage boundary in the source; never delete one to save time.

#pragma once

#if defined(G6_CLONE_ZONES)
#include "api/debug/kernel_profiler.hpp"
#define G6C_ZONE(name) DeviceZoneScopedN(name)
#else
#define G6C_ZONE(name)
#endif
