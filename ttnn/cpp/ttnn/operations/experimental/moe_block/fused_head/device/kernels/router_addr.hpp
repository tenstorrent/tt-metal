// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The router-core kernels' addresses the reference baked into defines (router_stream.RouterStream.pieces: `eng`, `sd`),
// here as common runtime args (the program hash then holds none). Every router kernel (NCRISC, compute, BRISC) gets
// the same two common args:
//   [0] the router cores' arena base (RouterStream.base - the CB range's address on the router cores)
//   [1] the engine-done semaphore address (RouterStream.arr[1]; 0 when the logits are on the router cores)
// and the host defines the switches / offsets:
//   RT_ENG_ON       (reference: RT_ENG = arr[1], only with the logits on the matmul cores)
//   RT_MSYNC_OFF    (reference: RT_MSYNC = base + off["msync"])
//   RT_PWORD_OFF    (reference: RT_PWORD = base + off["words"] + 128, the router split only)
//   RT_PART_OFF     (reference: RT_PART = base + off["part"], the router split only)
// The reference kernels test RT_ENG / RT_PART with #ifdef: they are defined here exactly when the reference defined
// them.
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
#ifdef RT_PWORD_OFF
#define RT_PWORD (get_common_arg_val<uint32_t>(0) + RT_PWORD_OFF)
#endif
#ifdef RT_PART_OFF
#define RT_PART (get_common_arg_val<uint32_t>(0) + RT_PART_OFF)
#endif
