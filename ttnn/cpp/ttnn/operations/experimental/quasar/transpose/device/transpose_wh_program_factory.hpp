// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transpose_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/program_spec_artifacts.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

#include <span>
#include <vector>

namespace ttnn::prim::qsr {

// Metal 2.0 (ProgramSpecFactoryWithOwnedTensorsConcept) factory for the W<->H transpose path (tiled + row-major).
struct TransposeWHProgramFactory {
    // Transpose owns no scratch/config tensors; routes through the owned-tensors adapter all the same.
    static std::vector<tt::tt_metal::MeshTensor> get_owned_tensors(
        const TransposeParams& operation_attributes, const TransposeInputs& tensor_args, Tensor& output_tensor);

    static ttnn::device_operation::ProgramSpecArtifacts create_program_spec(
        const TransposeParams& operation_attributes,
        const TransposeInputs& tensor_args,
        Tensor& output_tensor,
        std::span<const tt::tt_metal::MeshTensor> owned_tensors);
};

}  // namespace ttnn::prim::qsr
