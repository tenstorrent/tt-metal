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
// (cache hit). These are used to rebuild ProgramRunParams cheaply on cache hits, where
// only buffer addresses (and per-core work splits) need to be re-applied.
struct ReduceMultiCoreWSharedVariables {
    // Cores in the same iteration order used during create().
    // Allows override_runtime_arguments to address per-node RTAs without re-deriving
    // the work distribution.
    std::vector<CoreCoord> cores;

    // Cached work-split data so override_runtime_arguments can re-emit per-node RTAs
    // without re-running split_work_to_cores.
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_rows_per_core_group_1 = 0;
    uint32_t num_rows_per_core_group_2 = 0;
    uint32_t Wt = 0;
};

// ReduceMultiCoreWProgramFactory: width-axis reduction across multiple cores.
//
// Migrated to the Metal 2.0 host API:
//   - Uses ProgramSpec / DataflowBufferSpec / KernelSpec / WorkUnitSpec
//   - Built via MakeProgramFromSpec() and parameterized via SetProgramRunParameters().
//
// The factory follows ttnn's ProgramFactoryConcept (create + override_runtime_arguments)
// because the device_operation framework does not yet have a first-class adapter for
// ProgramSpec-based factories. The Program returned from MakeProgramFromSpec() is wrapped
// into a CachedProgram alongside ReduceMultiCoreWSharedVariables.
struct ReduceMultiCoreWProgramFactory {
    using shared_variables_t = ReduceMultiCoreWSharedVariables;
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
