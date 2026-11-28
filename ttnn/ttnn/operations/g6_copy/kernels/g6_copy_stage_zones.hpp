// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Per-stage device zones of g6_copy (PERMANENT instrumentation, gated).
// A DeviceZoneScopedN executes (~51 cycles each) in every profiler build, and g6_copy's stage zones
// sit on the reader -> writer critical path of every block: on the 1-page cells they were the
// whole loss to stock ttnn.copy (Perf 1: (1,1) bf16 1100 -> 906 ns with the zones compiled out).
// So the zones compile to nothing unless the host defines G6C_STAGE_ZONES, which it does only when
// the environment variable G6_COPY_STAGE_ZONES=1 is set (g6_copy_program_descriptor.py). Set it to
// get the per-stage breakdown; leave it unset for timing.
#include "api/debug/kernel_profiler.hpp"

#if defined(G6C_STAGE_ZONES)
#define G6C_STAGE_ZONE(name) DeviceZoneScopedN(name)
#else
#define G6C_STAGE_ZONE(name)
#endif
