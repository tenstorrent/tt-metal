// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program.hpp>

#include "ttnn/device_operation.hpp"
#include "reduce_op_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Shared variables threaded between create() (cache miss) and override_runtime_arguments()
// (cache hit). The H factory splits work along W columns, so each core processes a slice
// of contiguous columns. The snapshot lets override_runtime_arguments re-emit per-core RTAs
// without re-running split_work_to_cores.
struct ReduceMultiCoreHSharedVariables {
    std::vector<CoreCoord> cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_cols_per_core_group_1 = 0;
    uint32_t num_cols_per_core_group_2 = 0;
    uint32_t Wt = 0;
    uint32_t Ht = 0;
    uint32_t HtWt = 0;
};

// ReduceMultiCoreHProgramFactory: height-axis reduction across multiple cores.
//
// Migrated to the Metal 2.0 host API. Sharded inputs aren't supported on this path
// (matches the W and HW factories' stance) — sharded reductions still go through the
// legacy Gen1 pipeline upstream.
struct ReduceMultiCoreHProgramFactory {
    using shared_variables_t = ReduceMultiCoreHSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ReduceParams& operation_attributes, const Tensor& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ReduceParams& operation_attributes,
        const Tensor& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
