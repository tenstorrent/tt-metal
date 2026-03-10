// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "prefill_moe_compute_device_operation_types.hpp"
#include "tt_metal/api/tt-metalium/program.hpp"
#include <tt-metalium/program_cache.hpp>

namespace ttnn::operations::experimental::prefill_moe_compute {

struct PrefillMoeComputeMeshFactory {
    struct shared_variables_t {
        // Kernel handles
        tt::tt_metal::KernelHandle reader_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        tt::tt_metal::KernelHandle compute_kernel_id;
        tt::tt_metal::KernelHandle dispatch_kernel_id;
        tt::tt_metal::KernelHandle combine_kernel_id;

        // Core grids
        CoreRangeSet compute_cores;
        CoreCoord dispatch_core;
        CoreCoord combine_core;

        uint32_t num_cores;
        uint32_t grid_x;
        uint32_t grid_y;

        // Fabric return (populated only when enable_fabric_return is true)
        bool enable_fabric_return = false;
        std::optional<tt::tt_metal::KernelHandle> return_kernel_id;
        std::optional<tt::tt_metal::KernelHandle> recv_kernel_id;
        std::optional<CoreCoord> return_core;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const operation_attributes_t& attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output_tensor);
};

}  // namespace ttnn::operations::experimental::prefill_moe_compute
