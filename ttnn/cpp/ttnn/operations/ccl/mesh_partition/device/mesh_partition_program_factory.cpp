// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_partition_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tuple>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
// TODO(nuked-op): removed include of deleted slicing op header
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn::operations::ccl {
namespace detail {
uint32_t get_cluster_axis_index(
    const ttnn::MeshDeviceView& mesh_view,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const MeshPartitionDeviceOperation::operation_attributes_t& operation_attributes) {
    return operation_attributes.cluster_axis.has_value()
               ? ((operation_attributes.cluster_axis.value() == 0) ? mesh_coordinate[0] : mesh_coordinate[1])
               : common::get_linearized_index(mesh_coordinate, mesh_view);
}
}  // namespace detail

// TODO(nuked-op slice): SliceOp-based helper removed (slice primitive nuked).

ttnn::device_operation::CachedProgram<MeshPartitionDeviceOperation::MeshPartition::shared_variables_t>
MeshPartitionDeviceOperation::MeshPartition::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    (void)operation_attributes;
    (void)mesh_coordinate;
    (void)tensor_args;
    (void)tensor_return_value;
    // TODO(nuked-op slice): mesh_partition delegated to the nuked slice primitive.
    TT_THROW("mesh_partition: slice device-op was nuked; create_at is not implemented");
}

void MeshPartitionDeviceOperation::MeshPartition::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    (void)cached_workload;
    (void)operation_attributes;
    (void)tensor_args;
    (void)tensor_return_value;
    // TODO(nuked-op slice): no-op (slice primitive nuked).
}

}  // namespace ttnn::operations::ccl
