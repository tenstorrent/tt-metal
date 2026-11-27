// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transpose_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

#include <tt-metalium/core_coord.hpp>

namespace ttnn::prim::qsr {

// Metal 2.0 (MetalV2FactoryConcept) factory for the sharded, tiled W<->H transpose path. The
// input/output shards already live in L1, so cb_in0/cb_out0 are borrowed-memory DFBs that alias the
// input/output tensor shard buffers; the reader/writer are trivial sharded stubs.
struct TransposeWHShardedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TransposeParams& operation_attributes, const TransposeInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim::qsr
