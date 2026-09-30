// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric_all_gather_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

#include <tt-metalium/global_semaphore.hpp>

namespace ttnn::operations::experimental::fabric_all_gather {

// Fixed geometry of one cached program. The slot base and the active extent are runtime values derived from it on
// every dispatch (host scalars) or on device (metadata tensors).
struct FabricAllGatherGeometry {
    uint32_t num_stripes = 0;       // A
    uint32_t stripe_pages_max = 0;  // B_max: pages of a full stripe, the output placement stride
    uint32_t pages_per_slot = 0;    // input pages of one batch slot (selected-batch gathers)
    uint32_t gather_dim_size = 0;   // local (input) extent of the gather dim
    uint32_t group_size = 0;        // G
    uint32_t pages_per_slab = 0;    // active pages per stripe per block-cyclic slab (metadata extent)
};

struct FabricAllGatherFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle sender_kernel_id{};
        tt::tt_metal::KernelHandle copy_reader_kernel_id{};
        tt::tt_metal::KernelHandle copy_writer_kernel_id{};
        bool has_ports = false;
        bool has_copy = false;
        tt::tt_metal::GlobalSemaphore ready_sem;
        tt::tt_metal::GlobalSemaphore arrival_sem;
        CoreRangeSet worker_core_range;
        FabricAllGatherGeometry geometry;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const FabricAllGatherParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const FabricAllGatherInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const FabricAllGatherParams& operation_attributes,
        const FabricAllGatherInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::operations::experimental::fabric_all_gather
