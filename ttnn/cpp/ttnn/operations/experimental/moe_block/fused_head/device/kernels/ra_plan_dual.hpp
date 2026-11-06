// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The routing planner (ra_plan.hpp) for a kernel: RA_PLAN_RUN(a, B). Token loops in one program (RAP_DUAL,
// mb_loop.hpp): one planner, its tokens' layout picked by the loop (an earlier loop's plans no token: RAPI_*).
// RA_PLAN_NONE: no instance on this kernel's cores - no planner code.
#pragma once

#ifdef RA_PLAN_NONE
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/ra_accessors.hpp"
#define RA_PLAN_RUN(a, B) ((void)(a), (void)(B))
#else
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/ra_plan.hpp"
#define RA_PLAN_RUN(a, B) ra::plan((a), (B))
#endif
