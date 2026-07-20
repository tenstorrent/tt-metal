// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transpose_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

#include <tt-metalium/core_coord.hpp>

namespace ttnn::prim::qsr {

// Metal 2.0 (MetalV2FactoryConcept) factory for the sharded, row-major H<->C transpose path.
// The input/output shards live in L1, so cb_in/cb_out are borrowed-memory DFBs aliasing the
// input/output tensor shard buffers; sticks are reshuffled across cores purely by NoC reads. The
// generic path uses a single reader (cb_in/cb_out self-loop); the special case splits the work
// across a reader + writer. The per-core NoC-coordinate / stick-offset lists are passed as runtime
// varargs (variable length, recovered on device from named args).
struct TransposeHCShardedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TransposeParams& operation_attributes, const TransposeInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim::qsr
