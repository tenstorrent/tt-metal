// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_reduce_create_qkv_heads_device_operation_types.hpp"

#include "ttnn/distributed/types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <optional>

namespace ttnn::experimental::prim {

struct AllReduceCreateQkvHeadsMeshWorkloadFactory {
    // Per-coord program build.  The GlobalSemaphore lives on
    // AllReduceCreateQkvHeadsParams (allocated by the caller) and is excluded
    // from the program-cache key; override_runtime_arguments refreshes its
    // address and all tensor buffer addresses on every program-cache hit.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AllReduceCreateQkvHeadsParams& operation_attributes,
        const AllReduceCreateQkvHeadsInputs& tensor_args,
        AllReduceCreateQkvHeadsResult& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const AllReduceCreateQkvHeadsParams& operation_attributes,
        const AllReduceCreateQkvHeadsInputs& tensor_args,
        AllReduceCreateQkvHeadsResult& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
};

}  // namespace ttnn::experimental::prim
