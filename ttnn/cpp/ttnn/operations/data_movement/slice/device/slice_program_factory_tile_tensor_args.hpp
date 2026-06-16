// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

// Legacy ProgramDescriptor factory for the TILE tensor-args (use_tensor_args) path.
//
// Retained (NOT deleted) because ccl/mesh_partition's std::visit over slice's program_factory_t
// instantiates `Factory::create_descriptor(...)` for every variant alternative (see
// mesh_partition_program_factory.cpp), so this descriptor entry point must keep existing to keep
// mesh_partition building. As with the non-strided TILE path, a single factory struct cannot
// satisfy BOTH ProgramDescriptorFactoryConcept and ProgramSpecFactoryConcept, so slice's own
// Metal 2.0 path lives on the separate SliceTileTensorArgsSpecProgramFactory below; this struct
// stays descriptor-only for mesh_partition's reuse.
struct SliceTileTensorArgsProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

// Metal 2.0 (ProgramSpec) factory for the TILE tensor-args path. This is the factory slice's own
// SliceDeviceOperation dispatches through (select_program_factory routes the use_tensor_args path
// here). It produces the immutable ProgramSpec + mutable ProgramRunArgs and points at the forked
// *_m2 reader kernel (the writer reuses the shared writer_unary_interleaved_start_id_m2.cpp fork).
struct SliceTileTensorArgsSpecProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim
