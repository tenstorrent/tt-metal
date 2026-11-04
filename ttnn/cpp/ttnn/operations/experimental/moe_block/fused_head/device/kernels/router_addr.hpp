// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The router-core kernels' addresses, as common runtime args (the program hash then holds none). Every router kernel
// (NCRISC, compute, BRISC) gets the same two common args:
//   [0] the router cores' arena base (the CB range's address on the router cores)
//   [1] the engine-done semaphore address (0 when the logits are on the router cores)
// and the host defines the switches / offsets the macros below build the addresses from:
//   RT_ENG_ON       RT_ENG = common arg 1, only with the logits on the matmul cores
//   RT_MSYNC_OFF    RT_MSYNC = arena base + RT_MSYNC_OFF
// The kernels test RT_ENG with #ifdef: it is defined exactly when its switch is set.
#pragma once

// (get_common_arg_val must be declared before router_rows.hpp, which the kernels include ahead of their API headers)
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#else
#include "api/dataflow/dataflow_api.h"
#endif

#ifdef RT_ENG_ON
#define RT_ENG (get_common_arg_val<uint32_t>(1))
#endif
#ifdef RT_MSYNC_OFF
#define RT_MSYNC (get_common_arg_val<uint32_t>(0) + RT_MSYNC_OFF)
#endif
