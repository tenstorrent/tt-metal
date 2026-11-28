// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "embedding_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::prim::bringup {

struct EmbeddingsFusedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const EmbeddingParams& operation_attributes, const EmbeddingInputs& tensor_args, Tensor& tensor_return_value);
};

// NC.9 server order (EmbeddingParams::clamp_max set): the fused program a mesh row (its row index a reader arg), the
// per-call values (EmbeddingParams::server_start / server_valid) the reader's common runtime args, rewritten on every
// program-cache hit with the io tensors.
struct EmbeddingsServerOrderProgramFactory {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const EmbeddingParams& operation_attributes,
        const EmbeddingInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    static tt::tt_metal::ProgramRunArgs override_runtime_arguments(
        const EmbeddingParams& operation_attributes,
        const EmbeddingInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRange& range);
};

}  // namespace ttnn::prim::bringup
