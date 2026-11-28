// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/runtime_args_data.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "chunk_program_device_operation_types.hpp"

namespace ttnn::operations::chunk_program {

// A runtime-arg slot a call rewrites: io tensor ``src``'s address, or (src >= kValue) per-call value src - kValue;
// ``common``: the kernel's common runtime args (``core`` unused).
constexpr uint32_t kValue = 1u << 16;
struct SlotRef {
    uint32_t kernel = 0;
    CoreCoord core;
    uint32_t index = 0;
    uint32_t src = 0;
    bool common = false;
};

// One program of a mesh range (the Python _build loop body for that range) and the slots of its per-call args.
struct ChunkProgramProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ChunkProgramParams& attrs,
        const ChunkProgramInputs& tensor_args,
        const ttnn::MeshCoordinate& coord,
        std::vector<SlotRef>& slots);
};

// One program per range: the whole mesh (PAGE_COPY), an SP row (GLUED_ROWS, IDS_REORDER, GLUED_IDS) or a chip
// (DERIVE, OWNED_SEND), as the Python MeshProgramDescriptor. A hit writes the io addresses and the per-call values
// into every program's slots.
struct ChunkProgramMeshWorkloadFactory {
    struct Patch {
        tt::tt_metal::RuntimeArgsData* args = nullptr;
        uint32_t index = 0;
        uint32_t src = 0;
    };
    struct shared_variables_t {
        std::vector<Patch> patches;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const ChunkProgramParams& attrs,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const ChunkProgramInputs& tensor_args,
        Tensor& output);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const ChunkProgramParams& attrs,
        const ChunkProgramInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::operations::chunk_program
