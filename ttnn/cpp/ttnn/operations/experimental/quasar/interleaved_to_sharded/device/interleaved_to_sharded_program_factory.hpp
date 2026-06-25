// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/program_spec_artifacts.hpp"
#include "interleaved_to_sharded_op_types.hpp"

#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <span>
#include <vector>

namespace ttnn::prim::qsr {

// Metal 2.0 (ProgramSpecFactoryWithOwnedTensorsConcept) factory for interleaved->sharded.
struct InterleavedToShardedProgramFactory {
    // No scratch/config tensors; routes through the owned-tensors adapter for parity with other spec ops.
    static std::vector<tt::tt_metal::MeshTensor> get_owned_tensors(
        const InterleavedToShardedParams& operation_attributes,
        const InterleavedToShardedInputs& tensor_args,
        Tensor& output_tensor);

    static ttnn::device_operation::ProgramSpecArtifacts create_program_spec(
        const InterleavedToShardedParams& operation_attributes,
        const InterleavedToShardedInputs& tensor_args,
        Tensor& output_tensor,
        std::span<const tt::tt_metal::MeshTensor> owned_tensors);
};

}  // namespace ttnn::prim::qsr
