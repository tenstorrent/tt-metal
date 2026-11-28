// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/runtime_args_data.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "fabric_reduce_scatter_device_operation_types.hpp"
#include "fabric_reduce_scatter_planner.hpp"

namespace ttnn::operations::fabric_reduce_scatter {

// The generic op's constants.
constexpr uint32_t kCbBytes = 128 * 1024;  // per-CB batch budget: 2 x group chunks
constexpr uint32_t kMaxGroup = 8;          // chunks per batch, at most
constexpr uint32_t kIncEvery = 8;          // a fused write + increment once per this many chunks

// Per-mesh state: the plan and the three counters (A forward arrivals, B backward arrivals, ready), created
// once on the cores the plan uses, before the call allocates any tensor, then the mesh synchronized (every
// counter zero before any chip's first call). Held in a function-local ttsl::Indestructible map keyed by
// (mesh id, mesh shape, the plan's core set). Their addresses stay as create_descriptor wrote them.
struct MeshState {
    const plan::Plan* plan = nullptr;
    tt::tt_metal::GlobalSemaphore sem_a;
    tt::tt_metal::GlobalSemaphore sem_b;
    tt::tt_metal::GlobalSemaphore ready;
    uint32_t sem_a_addr = 0;
    uint32_t sem_b_addr = 0;
    uint32_t ready_addr = 0;
};
const MeshState& get_mesh_state(tt::tt_metal::distributed::MeshDevice& mesh_device, const plan::PlanInputs& inputs);

plan::PlanInputs plan_inputs(const FabricReduceScatterParams& attrs);

// One chip's program (Python create_mesh_program_descriptor's loop body for `coord`).
struct FabricReduceScatterProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const FabricReduceScatterParams& attrs,
        const FabricReduceScatterInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

// Owns the workload: one program per coordinate from create_descriptor (chips place their ports on
// different cores, so their programs do not merge), and a hit writes the call's addresses once into every
// program's common args.
struct FabricReduceScatterMeshWorkloadFactory {
    using descriptor_factory_t = FabricReduceScatterProgramFactory;

    struct shared_variables_t {
        // Common args of each kernel that holds a tensor address (kernel_index order).
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> port_reader;
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> port_sender;
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> final_reader;
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> final_writer;
        // (input, scratch, output) addresses the last completed hit wrote; empty until the first hit. Kept on
        // the first program's entry only: every program gets the same values.
        std::vector<uint32_t> applied_addresses;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const FabricReduceScatterParams& attrs,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const FabricReduceScatterInputs& tensor_args,
        Tensor& output);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const FabricReduceScatterParams& attrs,
        const FabricReduceScatterInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::operations::fabric_reduce_scatter
