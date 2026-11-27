// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/workload/legacy/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "sparse_kv_dispatch_device_operation_types.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

struct SparseKvDispatchProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SparseKvDispatchParams& params,
        const SparseKvDispatchInputs& tensor_args,
        Tensor& output_tensor,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const SparseKvDispatchParams& params,
        const SparseKvDispatchInputs& tensor_args,
        Tensor& output_tensor,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

// One program for the whole mesh: the per-chip descriptors (SparseKvDispatchProgramFactory) place every kernel, CB and
// semaphore on the same cores on every chip and differ only in runtime-arg values (the chip's index, its neighbours,
// its fabric connections, which directions it has), so they merge into one program with per-device runtime args.
// The common runtime args (tensor addresses, semaphores, cache slot) are the same on every chip and are patched on a
// cache hit.
struct SparseKvDispatchMeshWorkloadFactory {
    struct shared_variables_t {};
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const SparseKvDispatchParams& params,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const SparseKvDispatchInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const SparseKvDispatchParams& params,
        const SparseKvDispatchInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::operations::experimental::sparse_kv_push
