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

// Legacy ProgramDescriptor factory for the ROW_MAJOR HEIGHT-sharded in/out path.
//
// Retained (NOT deleted) because ccl/mesh_partition's std::visit over slice's
// program_factory_t instantiates Factory::create_descriptor(...) for EVERY variant
// alternative (mesh_partition_program_factory.cpp), so removing create_descriptor would
// break mesh_partition's build regardless of whether it ever selects this factory.
//
// A single factory struct cannot satisfy BOTH ProgramDescriptorFactoryConcept (has
// create_descriptor) and ProgramSpecFactoryConcept (has create_program_spec) — the
// AllFactoriesValid check requires each program_factory_t alternative to satisfy EXACTLY
// one. So slice's own Metal 2.0 path lives on the separate SliceRmShardedSpecProgramFactory
// below; this struct stays descriptor-only for mesh_partition's reuse.
struct SliceRmShardedProgramFactory {
    // Contract (1): per-coord ProgramDescriptor.  Both CBs are sharded
    // (CBDescriptor::buffer bound to input/output buffers); the framework
    // patches the dynamic CB addresses on cache hit via
    // apply_descriptor_runtime_args.  CB total_size/page_size are NOT patched
    // — padded_shape is folded into compute_program_hash() so each unique
    // sizing gets its own cache entry.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

// Metal 2.0 (ProgramSpec) factory for the ROW_MAJOR HEIGHT-sharded in/out path. This is the
// factory slice's own SliceDeviceOperation dispatches through. Case-2 (bridge) port: the reader
// keeps its hand-rolled NoC walk over a host-computed physical core-coordinate map; the input /
// output shard L1 base addresses flow through two borrowed-memory DFBs (borrowed_from src / dst),
// matching the legacy borrowed CBs. Points at the forked *_m2 reader kernel.
struct SliceRmShardedSpecProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim
