// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transpose_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

#include <tt-metalium/core_coord.hpp>

namespace ttnn::prim::qsr {

// Metal 2.0 (MetalV2FactoryConcept) factory for the sharded, row-major W<->H transpose path.
// The input/output shards live in L1, so cb_in0/cb_out0 are borrowed-memory DFBs aliasing the
// input/output tensor shard buffers; the reader gathers rows into a tile-staging DFB, the compute
// kernel tilizes + transposes + pack-untilizes, and (only when ht>8) a writer drains a staging DFB
// into the output shard.
struct TransposeWHShardedRMProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TransposeParams& operation_attributes, const TransposeInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim::qsr
