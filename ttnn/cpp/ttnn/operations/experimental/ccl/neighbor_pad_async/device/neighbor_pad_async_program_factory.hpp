// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "neighbor_pad_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/buffer.hpp>

namespace ttnn::experimental::prim {

// Artifacts returned by build_np_fabric_only_program_artifacts() — kernel handles
// needed for runtime arg updates and program caching.
struct NpFabricOnlyArtifacts {
    tt::tt_metal::KernelHandle h_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle h_writer_kernel_id = 0;
    tt::tt_metal::KernelHandle w_reader_kernel_id = 0;  // 0 if not 2D
    tt::tt_metal::KernelHandle w_writer_kernel_id = 0;  // 0 if not 2D
    bool has_w_fabric = false;
    tt::tt_metal::CoreRangeSet fabric_core_range;  // all NP cores (H + W if 2D)
};

// Add fabric-only NP kernels to an existing program.
// This is the NP equivalent of build_all_gather_async_minimal_default_program_artifacts().
// Can be called standalone (from NeighborPadAsync create_at) or from a fused op.
//
// Shape-derived parameters:
//   input_halo_dim_size:  input_tensor_shape[np_dim]
//   num_sticks_per_halo_dim: product of input_tensor_shape[np_dim+1 .. rank-2]
//   outer_dim_size:       product of input_tensor_shape[0 .. np_dim-1]
//   output_halo_dim_size: for fabric_only = padding_left + padding_right;
//                         for non-fabric_only = output_tensor_shape[np_dim]
//   fabric_only:          when true, skips local copy and uses compact halo buffer layout
NpFabricOnlyArtifacts build_np_fabric_only_program_artifacts(
    tt::tt_metal::Program& program,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const std::optional<ttnn::MeshCoordinate>& forward_coord,
    const std::optional<ttnn::MeshCoordinate>& backward_coord,
    uint32_t device_index,
    tt::tt_metal::Buffer* input_buffer,
    tt::tt_metal::Buffer* halo_buffer,
    // NP params
    uint32_t np_dim,  // which dimension to pad (1 for H in BTHWC)
    uint32_t padding_left,
    uint32_t padding_right,
    const std::string& padding_mode,
    uint32_t ring_size,
    ttnn::ccl::Topology topology,
    const GlobalSemaphore& h_neighbor_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    uint32_t num_links,
    // Shape-derived params
    uint32_t input_halo_dim_size,
    uint32_t num_sticks_per_halo_dim,
    uint32_t outer_dim_size,
    uint32_t output_halo_dim_size,
    bool fabric_only,
    tt::DataFormat data_format,
    // Optional 2D W-axis params (pad_dim2 != nullopt enables 2D mode)
    std::optional<uint32_t> pad_dim2,
    uint32_t pad2_left,
    uint32_t pad2_right,
    std::optional<uint32_t> pad2_cluster_axis,
    uint32_t pad2_num_links,
    const GlobalSemaphore& w_neighbor_semaphore,
    // Pre-computed W-axis topology (ignored when pad_dim2 == nullopt)
    const std::optional<ttnn::MeshCoordinate>& w_forward_coord,
    const std::optional<ttnn::MeshCoordinate>& w_backward_coord,
    uint32_t w_device_index,
    // Progress semaphore for T-batch pipelining: H-writer signals per T-batch, W-reader signals once at end.
    uint32_t progress_sem_addr,                                           // 0 if not pipelining
    const std::vector<std::pair<uint32_t, uint32_t>>& reader_noc_coords,  // conv3d reader cores to signal
    uint32_t progress_t_batch_size = 0  // T-batches per H-writer signal (0 = no H-writer signaling)
);

struct NeighborPadAsyncSharedVariables {
    // H fabric (1 consolidated reader kernel, 1 consolidated writer kernel)
    tt::tt_metal::KernelHandle h_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle h_writer_kernel_id = 0;

    // Local copy (1 consolidated reader kernel, 1 consolidated writer kernel)
    tt::tt_metal::KernelHandle local_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle local_writer_kernel_id = 0;

    // W fabric (1 consolidated reader kernel, 1 consolidated writer kernel)
    tt::tt_metal::KernelHandle w_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle w_writer_kernel_id = 0;

    bool has_local_copy = false;
    bool has_w_fabric = false;
};

struct NeighborPadAsyncMeshWorkloadFactory {
    using shared_variables_t = NeighborPadAsyncSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const NeighborPadAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const NeighborPadAsyncInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const NeighborPadAsyncParams& operation_attributes,
        const NeighborPadAsyncInputs& tensor_args,
        Tensor& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const NeighborPadAsyncParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const NeighborPadAsyncInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
