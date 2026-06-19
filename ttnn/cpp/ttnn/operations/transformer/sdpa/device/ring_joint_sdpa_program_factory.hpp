// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_multi_core_with_workers_program_factory.hpp"

namespace ttnn::prim {

namespace detail {

struct RingJointSDPADescriptorAdapterOperation {
    using operation_attributes_t = RingJointSDPAParams;
    using tensor_args_t = RingJointSDPAInputs;
    using spec_return_value_t = RingJointSDPAResultSpec;
    using tensor_return_value_t = RingJointSDPAResult;
};

}  // namespace detail

struct RingJointSDPAProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const RingJointSDPAParams& args,
        const RingJointSDPAInputs& tensor_args,
        RingJointSDPAResult& output_tensors,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

struct RingJointSDPAMeshWorkloadFactory {
    using descriptor_adapter_t = ttnn::device_operation::MeshDeviceOperationAdapter<
        detail::RingJointSDPADescriptorAdapterOperation>::DescriptorMeshWorkloadAdapter<RingJointSDPAProgramFactory>;
    using cached_mesh_workload_t = typename descriptor_adapter_t::cached_mesh_workload_t;

    static cached_mesh_workload_t create_mesh_workload(
        const RingJointSDPAParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const RingJointSDPAInputs& tensor_args,
        RingJointSDPAResult& output_tensors);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const RingJointSDPAParams& args,
        const RingJointSDPAInputs& tensor_args,
        RingJointSDPAResult& output_tensors);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<RingJointSDPAMeshWorkloadFactory>);

}  // namespace ttnn::prim
