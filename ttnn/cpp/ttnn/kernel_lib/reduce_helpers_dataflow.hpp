// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "llk_defs.h"
#include <tt-metalium/constants.hpp>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"

/**
 * @file reduce_helpers_dataflow.hpp
 * @brief Materialization of host-planned reduction auxiliary tiles.
 *
 * The public dataflow interface consumes one aggregate descriptor per planning
 * unit. The descriptor is appended independently of the compute call list and
 * already contains both the auxiliary CB ID and the physical tile recipe.
 */

namespace dataflow_kernel_lib {

/**
 * @brief Materialize and push one sequence-level auxiliary CB recipe.
 *
 * The planner aggregates the physical tiles needed by all calls in one
 * planning unit. The dataflow kernel receives this recipe independently from
 * the compute call list. Call this function exactly once at the beginning of
 * that unit's dataflow work; it fills and pushes the aggregate CB once, and all
 * compute calls use slices of those same tiles. Do not loop over the compute
 * call count or try to infer a call's partial mode from this physical recipe.
 *
 * @code{.cpp}
 * // AUXILIARY_ARGS_OFFSET follows this kernel's own CTA prefix.
 * using Auxiliary = ttnn::kernel_lib::ReduceAuxiliaryArgs<AUXILIARY_ARGS_OFFSET>;
 * dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<Auxiliary>();
 * @endcode
 *
 * For consecutive planning units, the next descriptor begins at
 * Auxiliary::next_compile_time_args_offset().
 *
 * @tparam Auxiliary A ReduceAuxiliaryArgs view of exactly one planning unit.
 */
template <typename Auxiliary>
FORCE_INLINE void prepare_reduce_auxiliary_tiles();

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl"
