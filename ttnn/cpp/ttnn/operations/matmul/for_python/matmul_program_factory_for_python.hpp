// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <variant>

#include "ttnn/operations/matmul/for_python/matmul_multicore_program_factory.hpp"
#include "ttnn/operations/matmul/for_python/matmul_multicore_reuse_optimized_program_factory.hpp"
#include "ttnn/operations/matmul/for_python/matmul_multicore_reuse_mcast_1d_program_factory.hpp"
#include "ttnn/operations/matmul/for_python/matmul_multicore_reuse_mcast_2d_program_factory.hpp"
#include "ttnn/operations/matmul/for_python/matmul_multicore_reuse_mcast_dram_sharded_program_factory.hpp"
#include "ttnn/operations/matmul/for_python/matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.hpp"

namespace ttnn::for_python {

using MatmulProgramFactory = std::variant<
    MatmulMultiCoreProgramFactory,
    MatmulMultiCoreReuseOptimizedProgramFactory,
    MatmulMultiCoreReuseMcast1DProgramFactory,
    MatmulMultiCoreReuseMcast2DProgramFactory,
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory,
    MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory>;

// gather_in0 selects a MeshWorkload factory that never had a create_descriptor entry point,
// so it is rejected here instead of being returned.
MatmulProgramFactory select_matmul_program_factory(
    const ttnn::prim::MatmulParams& operation_attributes, const ttnn::prim::MatmulInputs& tensor_args);

}  // namespace ttnn::for_python
