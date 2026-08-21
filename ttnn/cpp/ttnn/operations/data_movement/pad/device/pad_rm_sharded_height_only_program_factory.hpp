// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "pad_device_operation_types.hpp"

namespace ttnn::prim {

struct PadRmShardedHeightOnlyProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& tensor_return_value);

    // Every per-core arg is pinned by the hashed shapes and shard specs; only the two sharded CB base
    // addresses vary per dispatch. Replaces get_dynamic_runtime_args (#48928).
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const PadParams& operation_attributes,
        const PadInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};
}  // namespace ttnn::prim
