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

    // Patches ALL per-dispatch state (seed, src/dst addresses) into the cached program on every cache
    // hit -- in place, no descriptor rebuild. seed is hash-excluded (per-device offset applied when
    // use_per_device_seed); supersedes get_dynamic_runtime_args and resolve_bindings.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const DropoutParams& operation_attributes,
        const DropoutInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
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

    // Delegates to DropoutProgramFactory: both factories emit the same per-core layout, so one
    // implementation keeps the cache-hit patch from drifting between them.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const DropoutParams& operation_attributes,
        const DropoutInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
