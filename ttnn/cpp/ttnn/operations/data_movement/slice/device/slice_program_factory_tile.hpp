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

// Legacy ProgramDescriptor factory for the non-strided TILE path.
//
// Retained (NOT deleted) because ccl/mesh_partition builds its own MeshWorkload by
// directly calling SliceTileProgramFactory::create_descriptor() per mesh coordinate
// (see mesh_partition_program_factory.cpp). mesh_partition needs a ProgramDescriptor it
// can stamp into a Program and re-apply runtime args onto, so this descriptor entry point
// must keep existing.
//
// A single factory struct cannot satisfy BOTH ProgramDescriptorFactoryConcept (has
// create_descriptor) and ProgramSpecFactoryConcept (has create_program_spec) — the
// AllFactoriesValid check requires each program_factory_t alternative to satisfy EXACTLY
// one. So slice's own Metal 2.0 path lives on the separate SliceTileSpecProgramFactory
// below; this struct stays descriptor-only for mesh_partition's reuse.
struct SliceTileProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

// Metal 2.0 (ProgramSpec) factory for the non-strided TILE path. This is the factory
// slice's own SliceDeviceOperation dispatches through (select_program_factory routes the
// TILE path here). It produces the immutable ProgramSpec + mutable ProgramRunArgs and
// points at the forked *_m2 kernels (which use the experimental named-arg mechanism).
struct SliceTileSpecProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim
