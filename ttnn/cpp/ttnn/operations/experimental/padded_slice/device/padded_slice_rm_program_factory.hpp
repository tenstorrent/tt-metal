// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "padded_slice_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::experimental::prim {

struct PaddedSliceRMProgramFactory {
    // Metal 2.0 CustomProgramSpecFactoryConcept: one ProgramSpec stamped across the mesh.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PaddedSliceParams& operation_attributes, const PaddedSliceInputs& tensor_args, Tensor& output);

    // Cache-hit hook: refreshes the input and output tensor bindings, which is what the program needs
    // between dispatches — every other argument follows from the slice bounds and the output sharding,
    // both part of the program-cache key.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const PaddedSliceParams& operation_attributes,
        const PaddedSliceInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
};

}  // namespace ttnn::experimental::prim
