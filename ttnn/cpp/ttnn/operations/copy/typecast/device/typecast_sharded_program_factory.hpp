// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "typecast_device_op_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

// Descriptor-based factory for the sharded fast path of `ttnn::typecast`.
//
// Satisfies `ProgramDescriptorFactoryConcept` (defined in
// ttnn/api/ttnn/operation_concepts.hpp): exposes a single
// `create_descriptor()` and declares neither `cached_program_t` nor
// `override_runtime_arguments`. The framework's
// `DescriptorMeshWorkloadAdapter` (mesh_device_operation_adapter.hpp) builds
// the cached Program from the returned descriptor on cache miss and, on cache
// hit, re-invokes `create_descriptor()` and calls `apply_descriptor_runtime_args`
// to refresh the dynamic CB addresses recorded on each `CBDescriptor::buffer`.
//
// The other three factories in `TypecastDeviceOperation::program_factory_t`
// (`TypecastProgramFactory`, `TypecastSubgridProgramFactory`,
// `TypecastRowMajorChunkedProgramFactory`) are still on the legacy
// `ProgramFactoryConcept` path. Per-alternative concept dispatch in
// `dispatch_to_mesh_workload_factory` (device_operation.hpp) lets the two
// styles coexist within the same `std::variant` without any change to
// `TypecastDeviceOperation` itself.
struct TypecastShardedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim
