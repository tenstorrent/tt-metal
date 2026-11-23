// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sparse_kv_dispatch with the fused plan (kvd4): the plan on a RISC the dispatch has no kernel for (the spare cores'
// BRISC and NCRISC, the planners' NCRISC). sparse_push_plan.cpp, built with PLAN_FUSED.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/plan/device/kernels/sparse_push_plan.cpp"

void kernel_main() { plan_prologue(); }
