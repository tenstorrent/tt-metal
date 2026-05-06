// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program.hpp>

#include "ttnn/device_operation.hpp"
#include "reduce_op_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Shared variables threaded between create() (cache miss) and override_runtime_arguments()
// (cache hit). Single-core HW reduction has no work split, so we only need to remember
// which core was used so override_runtime_arguments can re-emit RTAs at the right node.
struct ReduceSingleCoreHwSharedVariables {
    CoreCoord core;
    uint32_t num_tensor_tiles = 0;
    uint32_t out_dim_divider = 0;  // Ht * Wt
};

// ReduceSingleCoreHwProgramFactory: full HW-axis reduction on a single core.
//
// Migrated to the Metal 2.0 host API:
//   - Uses ProgramSpec / DataflowBufferSpec / KernelSpec / WorkUnitSpec
//   - Built via MakeProgramFromSpec() and parameterized via SetProgramRunParameters().
struct ReduceSingleCoreHwProgramFactory {
    using shared_variables_t = ReduceSingleCoreHwSharedVariables;
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
