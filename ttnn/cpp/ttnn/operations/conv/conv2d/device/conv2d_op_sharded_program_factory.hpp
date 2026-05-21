// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::prim {

// Descriptor-based factory for the HEIGHT_SHARDED / BLOCK_SHARDED variants of
// conv2d (everything that is not WIDTH_SHARDED).
//
// Satisfies `ProgramDescriptorFactoryConcept` via the declarative
// `create_workload_descriptor` contract (contract 2 in
// mesh_device_operation_adapter.hpp).  The declarative contract is required
// because this factory allocates a workload-scoped reader-indices config
// Tensor whose `Buffer*` is referenced from the per-coord program's CB /
// compile-time args; the Tensor must outlive the cached MeshWorkload, which
// `WorkloadDescriptor::buffers` arranges via shared-ptr ownership.
//
// The sibling variant `Conv2dWidthShardedProgramFactory` is also on the
// descriptor path; per-alternative concept dispatch in
// `dispatch_to_mesh_workload_factory` (device_operation.hpp) lets both styles
// live in the same `Conv2dDeviceOperation::program_factory_t` variant
// without any change to `Conv2dDeviceOperation` itself.
struct Conv2dShardedProgramFactory {
    // Builds the entire workload in one call (invoked ONCE per workload on
    // cache miss):
    //   1. Allocates the sliding-window reader-indices config Tensor on the
    //      input tensor's device and parks it in `buffers` so the framework
    //      keeps it alive for the cached MeshWorkload's lifetime — the
    //      reader/writer kernels reference this buffer either via a globally-
    //      allocated CB (L1 path) or via compile-time args
    //      (config_tensors_in_dram path).
    //   2. Builds one `ProgramDescriptor` and replicates it across all
    //      coordinate ranges of `tensor_coords`.
    //
    // On cache hits the framework's `BufferBinding` fast path patches the
    // weight / bias buffer addresses recorded via
    // `KernelDescriptor::emplace_runtime_args(Buffer*)` and the dynamic CB
    // addresses recorded on `CBDescriptor::buffer` — `create_workload_descriptor`
    // is not re-invoked.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const Conv2dParams& operation_attributes,
        const Conv2dInputs& tensor_args,
        Tensor& output_tensor,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::prim
