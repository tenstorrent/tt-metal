// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program.hpp>

#include "ttnn/device_operation.hpp"
#include "welford_reduce_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Shared variables threaded between create() (cache miss) and override_runtime_arguments()
// (cache hit). Welford's work split varies by reduce_dim:
//   - W-reduce: work units = NC * Ht (one per row of tiles)
//   - H-reduce: work units = NC * Wt (one per column of tiles)
//   - HW-reduce: work units = NC / reduce_batch_size (one per output scalar)
// Each core gets a slice of work units; per-core RTAs include the reader's tile-id
// math which differs per reduce_dim.
struct WelfordReduceSharedVariables {
    std::vector<CoreCoord> cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_work_units_per_core_group_1 = 0;
    uint32_t num_work_units_per_core_group_2 = 0;

    // Tensor metadata captured at create() so override_runtime_arguments can rebuild
    // per-core tile-id math without re-reading the tensor.
    uint32_t Wt = 0;
    uint32_t Ht = 0;
    uint32_t HtWt = 0;
    uint32_t reduce_batch_size = 1;

    // The reduce_dim is also captured because the per-core RTA shape depends on it.
    tt::tt_metal::ReduceOpDim reduce_dim = tt::tt_metal::ReduceOpDim::W;
};

// WelfordReduceProgramFactory: variance / std-dev reduction (Welford's online algorithm)
// across one of W, H, HW. Migrated to the Metal 2.0 host API.
struct WelfordReduceProgramFactory {
    using shared_variables_t = WelfordReduceSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const WelfordReduceParams& operation_attributes, const Tensor& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const WelfordReduceParams& operation_attributes,
        const Tensor& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
