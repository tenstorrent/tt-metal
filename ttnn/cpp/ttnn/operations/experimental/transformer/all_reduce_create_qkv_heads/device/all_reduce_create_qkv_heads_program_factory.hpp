// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_reduce_create_qkv_heads_device_operation_types.hpp"

#include "ttnn/distributed/types.hpp"

#include <tt-metalium/program_descriptors.hpp>

#include <optional>

namespace ttnn::experimental::prim {

struct AllReduceCreateQkvHeadsMeshWorkloadFactory {
    // Per-coord program build.  The GlobalSemaphore lives on
    // AllReduceCreateQkvHeadsParams (allocated by the caller), so no
    // prepare_resources hook is required -- the semaphore is passed through and
    // its address is written into runtime args every dispatch via the normal
    // apply_descriptor_runtime_args path.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AllReduceCreateQkvHeadsParams& operation_attributes,
        const AllReduceCreateQkvHeadsInputs& tensor_args,
        AllReduceCreateQkvHeadsResult& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
