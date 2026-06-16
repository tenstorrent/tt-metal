// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim {

// Legacy ProgramDescriptor factory for the ROW_MAJOR strided (step != 1) path.
//
// Retained (NOT deleted) because ccl/mesh_partition's std::visit over slice's
// program_factory_t instantiates Factory::create_descriptor(...) for EVERY variant alternative
// (mesh_partition_program_factory.cpp). A single struct cannot satisfy both the descriptor and
// the ProgramSpec factory concepts, so slice's Metal 2.0 path lives on the separate
// SliceRmStrideSpecProgramFactory below; this struct stays descriptor-only for mesh_partition.
struct SliceRmStrideProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

// Metal 2.0 (ProgramSpec) factory for the ROW_MAJOR strided path. This is the factory slice's own
// SliceDeviceOperation dispatches through. Case-1 port: src/dst are clean Buffer* address RTAs
// (no host-composed offset, no per-shard page-size override — the legacy kernels build their
// TensorAccessor with the binding's default aligned page size), so they re-express as ordinary
// TensorBindings. Selects between the 4D and ND forked *_m2 kernels at runtime by rank, exactly
// like create_descriptor.
struct SliceRmStrideSpecProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim
