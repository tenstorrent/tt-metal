// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dropout_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {
struct DropoutProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const DropoutParams& args, const DropoutInputs& tensor_args, Tensor& output);
};

struct DropoutMeshWorkloadFactory {
    // Dropout generates N different programs, but they differ only in the per-device seed set as a runtime argument.
    // TODO: when heterogeneous runtime arguments are supported, create a single program for all devices, and only
    // override the runtime arguments for each device. In addition, use `CachedMeshWorkload` instead of
    // `AdaptedCachedMeshWorkload`, as only a single `shared_variables_t` is needed.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const DropoutParams& args,
        const DropoutInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
