// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_device_operation_types.hpp"

namespace ttnn::prim {

// The SDPA worker grid: the program config's grid less its last column, which hosts the fabric MUX kernels.
inline CoreCoord exp_ring_sdpa_grid(CoreCoord user_grid) { return CoreCoord{user_grid.x - 1, user_grid.y}; }

namespace exp_ring_joint_sdpa_dynamic {
// Kernel indices — must match the desc.kernels push order (reader, writer, writer_fabric, compute[, mux]).
inline constexpr uint32_t kReaderKernelIdx = 0;
inline constexpr uint32_t kWriterFabricKernelIdx = 2;
// Per-core reader runtime-arg slot of the first per-link semaphore address; slots
// kReaderSemaphoreArgBase .. +num_links-1 hold args.semaphore[lnk].address().
// Preceded by 8 tensor addresses, 3 Q-work args, 10 chain/forwarding args, and num_links.
inline constexpr uint32_t kReaderSemaphoreArgBase = 22;
// Reader args after the semaphore addresses: ring_size, ring_index, direction, then the
// split-head forwarding dedup descriptor (dedup_role, buddy_injector_x, buddy_injector_y).
inline constexpr uint32_t kReaderTrailingArgCount = 6;
constexpr uint32_t reader_arg_count(uint32_t num_links) {
    return kReaderSemaphoreArgBase + num_links + kReaderTrailingArgCount;
}
// Per-core fabric-writer runtime-arg slot of out_ready_sem_addr (= args.semaphore[link].address()).
// Preceded by 2 output addresses, 3 Q-work args, 3 ring-indexer args, and 17 MUX args.
inline constexpr uint32_t kWriterFabricOutReadySemArg = 25;
// Fabric-writer args after out_ready_sem_addr: injector x/y, num_muxes, mux index, AG Wt/Ht,
// gathered k/v addresses, dedup_skip_forward (split-head forwarding dedup).
inline constexpr uint32_t kWriterFabricArgCount = kWriterFabricOutReadySemArg + 10;
}  // namespace exp_ring_joint_sdpa_dynamic

namespace detail {

struct ExpRingJointSDPADescriptorAdapterOperation {
    using operation_attributes_t = ExpRingJointSDPAParams;
    using tensor_args_t = ExpRingJointSDPAInputs;
    using spec_return_value_t = ExpRingJointSDPAResultSpec;
    using tensor_return_value_t = ExpRingJointSDPAResult;
};

}  // namespace detail

struct ExpRingJointSDPAProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const ExpRingJointSDPAParams& operation_attributes,
        const ExpRingJointSDPAInputs& tensor_args,
        ExpRingJointSDPAResult& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

struct ExpRingJointSDPAMeshWorkloadFactory {
    using descriptor_adapter_t =
        ttnn::device_operation::MeshDeviceOperationAdapter<detail::ExpRingJointSDPADescriptorAdapterOperation>::
            DescriptorMeshWorkloadAdapter<ExpRingJointSDPAProgramFactory>;
    using cached_mesh_workload_t = typename descriptor_adapter_t::cached_mesh_workload_t;

    static cached_mesh_workload_t create_mesh_workload(
        const ExpRingJointSDPAParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const ExpRingJointSDPAInputs& tensor_args,
        ExpRingJointSDPAResult& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const ExpRingJointSDPAParams& operation_attributes,
        const ExpRingJointSDPAInputs& tensor_args,
        ExpRingJointSDPAResult& tensor_return_value);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<ExpRingJointSDPAMeshWorkloadFactory>);

}  // namespace ttnn::prim
