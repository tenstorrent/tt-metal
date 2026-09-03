// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "llk_defs.h"
#include <tt-metalium/constants.hpp>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"

namespace dataflow_kernel_lib {

/**
 * @brief Materialize and push one sequence-level auxiliary CB recipe.
 *
 * The planner aggregates the physical tiles needed by all calls in one
 * planning unit. The dataflow kernel receives this recipe independently from
 * the compute call list.
 */
template <typename Auxiliary>
FORCE_INLINE void prepare_reduce_auxiliary_tiles();

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl"
