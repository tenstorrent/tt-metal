// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "all_gather_regime_a_matmul_async_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// Phase A (DRAM-staged) fused all-gather + regime_a_matmul program factory.
//
// This is a MULTI-DEVICE op: each participating device builds its own program (a fabric injector that gathers
// the in0 K-shards into a per-device DRAM gather buffer, plus a replicated regime_a compute engine that reads
// that gather buffer). The framework drives this through the MeshWorkload pattern: create_at() builds ONE
// device's program given its MeshCoordinate (from which we derive its K-shard index + fabric neighbours), and
// the framework fans create_at() over every coordinate in tensor_coords to assemble the MeshWorkload.
//
// D=1 never reaches here (the public op delegates to regime_a_matmul).
struct AllGatherRegimeAMatmulAsyncProgramFactory {
    struct shared_variables_t {
        // regime_a compute engine (split-NoC kernel handles, mirrors RegimeAMatmulProgramFactory).
        uint32_t num_cores{};
        std::vector<tt::tt_metal::CoreCoord> cores;  // logical worker coords, index i = bank*preaders + slice
        std::vector<uint32_t> core_noc;              // per-core NoC group (0 => A/g0, 1 => B/g1)
        tt::tt_metal::KernelHandle readerA{};
        tt::tt_metal::KernelHandle readerB{};
        tt::tt_metal::KernelHandle writerA{};
        tt::tt_metal::KernelHandle writerB{};
        tt::tt_metal::KernelHandle compute{};

        // Fabric injector (gathers in0 shards into the DRAM gather buffer).
        std::vector<tt::tt_metal::CoreCoord> injector_cores;
        tt::tt_metal::KernelHandle injector{};
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    // Build ONE device's program at the given mesh coordinate. The framework synthesizes create_mesh_workload
    // by fanning this over tensor_coords (see device_operation.hpp create_mesh_workload_from_workload_factory).
    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const AllGatherRegimeAMatmulAsyncParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const AllGatherRegimeAMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& output_tensors);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AllGatherRegimeAMatmulAsyncParams& operation_attributes,
        const AllGatherRegimeAMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& output_tensors);
};

}  // namespace ttnn::experimental::prim
