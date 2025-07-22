// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "multidevice_scatter_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device_pool.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "cpp/ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/fabric.hpp>
#include "ttnn/operations/data_movement/slice/device/slice_op.hpp"

namespace ttnn::operations::ccl {

MultiDeviceScatterDeviceOperation::MultiDeviceScatter::cached_mesh_workload_t
MultiDeviceScatterDeviceOperation::MultiDeviceScatter::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<MultiDeviceScatterDeviceOperation::MultiDeviceScatter::shared_variables_t>
MultiDeviceScatterDeviceOperation::MultiDeviceScatter::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    using namespace ttnn::ccl;

    const auto& input_tensor = tensor_args.input_tensor;
    auto mesh_device = input_tensor.mesh_device();
    const auto& mesh_view = mesh_device->get_view();
    const uint32_t ring_devices =
        (operation_attributes.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    TT_FATAL(ring_devices > 1, "reduce_scatter async op will only work for ring_devices > 1, but has {}", ring_devices);

    auto target_device = mesh_device->get_device(mesh_coordinate);

    std::vector<IDevice*> devices = (operation_attributes.cluster_axis == 0)
                                        ? mesh_view.get_devices_on_column(mesh_coordinate[1])
                                        : mesh_view.get_devices_on_row(mesh_coordinate[0]);
    uint32_t cluster_axis_size = devices.size();
    uint32_t cluster_index = operation_attributes.cluster_axis == 0 ? mesh_coordinate[0] : mesh_coordinate[1];
    auto input_shape = input_tensor.logical_shape();
    uint32_t dim = operation_attributes.dim;
    uint32_t rank = input_shape.size();

    auto scattered_dim_size = input_shape[dim] / cluster_axis_size;

    std::vector<uint32_t> begins(rank, 0);
    auto ends = input_shape;
    std::vector<uint32_t> strides(rank, 1);

    begins[dim] = cluster_index * scattered_dim_size;
    ends[dim] = begins[dim] + scattered_dim_size;

    auto slice_start = ttnn::Shape(begins);
    auto slice_step = ttnn::Shape(strides);

    log_info(
        tt::LogAlways,
        "Slice at ({}, {}) will have begins {}, ends {}, step {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        slice_start,
        ends,
        slice_step);

    auto slice_op = ttnn::operations::data_movement::SliceDeviceOperation{
        .slice_start = slice_start,
        .slice_end = ends,
        .step = slice_step,
        .output_mem_config = operation_attributes.output_mem_config};
    auto input_tensors = std::vector<ttnn::Tensor>{tensor_args.input_tensor};
    auto output_tensors = std::vector<ttnn::Tensor>{tensor_return_value};
    auto cached_program = slice_op.create_program(input_tensors, output_tensors);
    TT_FATAL(
        cached_program.override_runtime_arguments_callback.has_value(),
        "override_runtime_arguments_callback is not set");

    // -- building the return value -----------------------------------
    shared_variables_t vars{
        // if the optional holds a callback, move it; otherwise construct an empty
        // std::function (== “no-op”)
        .override_runtime_arguments_callback = cached_program.override_runtime_arguments_callback.value_or(
            OverrideRuntimeArgsCallback{}),  //   ^ empty functor
        .slice_op = slice_op};

    return {std::move(cached_program.program), std::move(vars)};
}

void MultiDeviceScatterDeviceOperation::MultiDeviceScatter::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    std::vector<tt::tt_metal::Tensor> input_tensors{tensor_args.input_tensor};
    std::vector<std::optional<const tt::tt_metal::Tensor>> input_tensor_options{};
    std::vector<tt::tt_metal::Tensor> output_tensors{tensor_return_value};
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);
        shared_variables.override_runtime_arguments_callback(
            (const void*)&shared_variables.slice_op, program, input_tensors, input_tensor_options, output_tensors);
    }
}

}  // namespace ttnn::operations::ccl
