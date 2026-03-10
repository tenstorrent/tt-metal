// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "neighbor_pad_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct NeighborPadAsyncSharedVariables {
    // H fabric (1 consolidated reader kernel, 1 consolidated writer kernel)
    tt::tt_metal::KernelHandle h_reader_kernel_id;
    tt::tt_metal::KernelHandle h_writer_kernel_id;
    std::vector<tt::tt_metal::CoreCoord> h_fabric_core_coords;

    // Local copy (1 consolidated reader kernel, 1 consolidated writer kernel)
    tt::tt_metal::KernelHandle local_reader_kernel_id;
    tt::tt_metal::KernelHandle local_writer_kernel_id;
    std::vector<tt::tt_metal::CoreCoord> local_copy_core_coords;

    // W fabric (1 consolidated reader kernel, 1 consolidated writer kernel)
    tt::tt_metal::KernelHandle w_reader_kernel_id;
    tt::tt_metal::KernelHandle w_writer_kernel_id;
    std::vector<tt::tt_metal::CoreCoord> w_fabric_core_coords;

    uint32_t num_links = 0;
    uint32_t num_directions = 0;
    uint32_t num_w_links = 0;
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
