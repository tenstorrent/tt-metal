// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "slice_write_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct SliceWriteRMInterleavedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SliceWriteParams& operation_attributes, const SliceWriteInputs& tensor_args, Tensor& tensor_return_value);

    // compute_program_hash keys this factory on the last-dim start only, so the outer-dim start offsets
    // baked into the writer's runtime args have to be recomputed on every cache hit.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const SliceWriteParams& operation_attributes,
        const SliceWriteInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
};

}  // namespace ttnn::experimental::prim
