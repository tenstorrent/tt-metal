// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// The compute thread this kernel is compiled for. On Quasar every thread's addrmods live in one config
// region indexed by thread id, so addr_mod_t::set() must know its own thread. This is the single source
// of truth for the thread id (0=unpack, 1=math, 2=pack, 3=isolate-SFPU), replacing the per-namespace
// TRISC_ID constants that used to live in c{unpack,math,pack}_common.h. It is derived from the
// -DCOMPILE_FOR_TRISC=<n> the build already bakes into every compute compilation (metal:
// llrt/hal/tt-2xx/hal_2xx_common.cpp; tt-llk tests: test_config.py) -- a compiler-provided macro, so it
// needs no other include and forms no cycle.
//
// This header is included only from ckernel_addrmod.h, which is compiled solely in compute (trisc)
// translation units. Keep it that way: it must NOT be pulled into data-movement/BRISC/NCRISC builds
// (e.g. dataflow reader/writer kernels), which do not define COMPILE_FOR_TRISC and would trip the guard.
#ifndef COMPILE_FOR_TRISC
#error "COMPILE_FOR_TRISC must be defined for compute (trisc) builds; the addrmod thread id derives from it"
#endif

namespace ckernel
{

// COMPILE_FOR_TRISC is the cluster-GLOBAL processor id (NEO_n_COMPUTE_m = n*4 + m, 0..15); the
// per-Neo thread id this indexes config by is that id modulo 4. See the ThreadId enum note in
// ckernel_defs.h, and the same reduction in ckernel.h's mailbox self-checks.
constexpr std::uint32_t TRISC_ID = COMPILE_FOR_TRISC % 4;

// Validates the define itself: the id must name a compute processor of the 4-Neo cluster. (Asserting
// TRISC_ID < 4 would be vacuous - the reduction above already guarantees it.)
static_assert(COMPILE_FOR_TRISC < 16, "COMPILE_FOR_TRISC must be a Quasar compute processor id in [0, 15]");

} // namespace ckernel
