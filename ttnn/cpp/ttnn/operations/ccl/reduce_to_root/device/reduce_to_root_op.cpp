// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <tt_stl/assert.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "reduce_to_root_op.hpp"

using namespace tt::tt_metal;
namespace ttnn::operations::ccl {

using cached_workload_t = device_operation::CachedProgram<ReduceToRootOp::ReduceToRoot::shared_variables_t>;

void ReduceToRootOp::validate(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_l = tensor_args.input_tensor_l;

    auto* mesh_device = input_l.device();

    const auto& optional_output_tensor_l = tensor_args.optional_output_tensor_l;
    const auto& optional_output_tensor_s = tensor_args.optional_output_tensor_s;
    const auto& optional_output_tensor_m = tensor_args.optional_output_tensor_m;
    if (optional_output_tensor_l.has_value()) {
        const auto output_spec = compute_output_specs(operation_attributes, tensor_args).at(1);

        TT_FATAL(
            output_spec[0] == optional_output_tensor_l.value().tensor_spec(),
            "Optional output tensor spec {} does not match computed output spec {}",
            optional_output_tensor_l.value().tensor_spec(),
            output_spec[0]);

        TT_FATAL(
            optional_output_tensor_l.value().device() == mesh_device,
            "Output tensor must be allocated on same mesh device as input tensor");
    }
    if (optional_output_tensor_s.has_value()) {
        const auto output_spec = compute_output_specs(operation_attributes, tensor_args).at(1);
        TT_FATAL(
            output_spec[1] == optional_output_tensor_s.value().tensor_spec(),
            "Optional output tensor spec {} does not match computed output spec {}",
            optional_output_tensor_s.value().tensor_spec(),
            output_spec[1]);

        TT_FATAL(
            optional_output_tensor_s.value().device() == mesh_device,
            "Output tensor must be allocated on same mesh device as input tensor");
    }
    if (optional_output_tensor_m.has_value()) {
        const auto output_spec = compute_output_specs(operation_attributes, tensor_args).at(1);
        TT_FATAL(
            output_spec[2] == optional_output_tensor_m.value().tensor_spec(),
            "Optional sparse output token tensor spec {} does not match computed output spec {}",
            optional_output_tensor_m.value().tensor_spec(),
            output_spec[2]);

        TT_FATAL(
            optional_output_tensor_m.value().device() == mesh_device,
            "Output tensor must be allocated on same mesh device as input tensor");
    }
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const uint32_t input_page_size_bytes = input_l.tensor_spec().compute_page_size_bytes();

    TT_FATAL(
        input_page_size_bytes % l1_alignment == 0 || input_page_size_bytes == l1_alignment,
        "Tensor page size must be aligned");
    // input mux cores should be 4
    if (operation_attributes.input_mux_cores.has_value()) {
        TT_FATAL(
            operation_attributes.input_mux_cores.value().size() == 4,
            "Input mux cores size must be 4, got {}",
            operation_attributes.input_mux_cores.value().size());
    }
};

ReduceToRootOp::spec_return_value_t ReduceToRootOp::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor_l = tensor_args.input_tensor_l;
    const auto& input_tensor_s = tensor_args.input_tensor_s;
    const auto& input_tensor_m = tensor_args.input_tensor_m;

    std::vector<TensorSpec> final_output_spec = {
        input_tensor_l.tensor_spec(), input_tensor_s.tensor_spec(), input_tensor_m.tensor_spec()};

    std::vector<TensorSpec> intermediate_specs;
    if (tensor_args.optional_intermediate_tensor.has_value()) {
        intermediate_specs.push_back(tensor_args.optional_intermediate_tensor.value().tensor_spec());
        return {intermediate_specs, final_output_spec};
    }
    // intermediate shape is the shape of the 3 tenssors combined so that we can send them all in a single packet
    uint32_t shape_0 = final_output_spec[0].memory_config().shard_spec()->shape[0];
    uint32_t shape_1 = final_output_spec[0].memory_config().shard_spec()->shape[1] +
                       (2 * final_output_spec[1].memory_config().shard_spec()->shape[1]);
    Shape intermediate_shape = Shape{shape_0, shape_1};
    TensorSpec intermediate_spec(intermediate_shape, final_output_spec[0].tensor_layout());
    intermediate_specs.push_back(intermediate_spec);

    return {intermediate_specs, final_output_spec};
}

ReduceToRootOp::tensor_return_value_t ReduceToRootOp::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(operation_attributes, tensor_args);

    auto* mesh_device = tensor_args.input_tensor_l.device();

    std::vector<ttnn::Tensor> intermediate_output_tensors;
    std::vector<ttnn::Tensor> final_output_tensors;

    auto intermediate_output_tensor_l = create_device_tensor(output_specs.at(0)[0], mesh_device);
    if (tensor_args.optional_intermediate_tensor.has_value()) {
        intermediate_output_tensor_l = tensor_args.optional_intermediate_tensor.value();
    }

    auto final_output_tensor_l = create_device_tensor(output_specs.at(1)[0], mesh_device);
    if (tensor_args.optional_output_tensor_l.has_value()) {
        final_output_tensor_l = tensor_args.optional_output_tensor_l.value();
    }

    auto final_output_tensor_s = create_device_tensor(output_specs.at(1)[1], mesh_device);
    if (tensor_args.optional_output_tensor_s.has_value()) {
        final_output_tensor_s = tensor_args.optional_output_tensor_s.value();
    }

    auto final_output_tensor_m = create_device_tensor(output_specs.at(1)[2], mesh_device);
    if (tensor_args.optional_output_tensor_m.has_value()) {
        final_output_tensor_m = tensor_args.optional_output_tensor_m.value();
    }

    intermediate_output_tensors = {intermediate_output_tensor_l};
    final_output_tensors = {final_output_tensor_l, final_output_tensor_s, final_output_tensor_m};

    return {intermediate_output_tensors, final_output_tensors};
}

ReduceToRootOp::ReduceToRoot::cached_mesh_workload_t ReduceToRootOp::ReduceToRoot::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor_l.device();
    auto sd_id = mesh_device->get_sub_device_ids().at(0);
    auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
    semaphores.reserve(2);
    // 2 semaphores: one for each round
    for (size_t i = 0; i < 2; ++i) {
        semaphores.push_back(ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0));
    }
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready in reduce_to_root op");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});
    log_debug(tt::LogOp, "Synchronize devices in reduce_to_root op done");

    const auto& coords = tensor_coords.coords();
    auto topology = tt::tt_fabric::Topology::Linear;
    for (const auto& coord : coords) {
        std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor_l, coord, 1, topology, std::nullopt);

        std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor_l, coord, -1, topology, std::nullopt);

        if (coord == operation_attributes.root_coord) {
            if (forward_coord.has_value() == 0 || backward_coord.has_value() == 0) {
                TT_FATAL(false, "Root device must have both forward and backward neighbors in reduce_to_root op");
            }
        }
        auto cached_workload = create_at(
            operation_attributes, coord, forward_coord, backward_coord, tensor_args, tensor_return_value, semaphores);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_workload.program));
        shared_variables.emplace(coord, std::move(cached_workload.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

cached_workload_t ReduceToRootOp::ReduceToRoot::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    std::optional<MeshCoordinate>& forward_coord,
    std::optional<MeshCoordinate>& backward_coord,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    std::vector<tt::tt_metal::GlobalSemaphore>& semaphores) {
    const auto& root_coordinate = operation_attributes.root_coord;
    const float scale_fp32 = operation_attributes.scale_fp32;

    return reduce_to_root_program_factory(
        tensor_args,
        operation_attributes,
        root_coordinate,
        scale_fp32,
        mesh_coordinate,
        forward_coord,
        backward_coord,
        tensor_return_value,
        semaphores);
}

}  // namespace ttnn::operations::ccl

namespace ttnn::prim {
ttnn::operations::ccl::ReduceToRootOp::tensor_return_value_t reduce_to_root(
    const Tensor& input_tensor_l,
    const Tensor& input_tensor_s,
    const Tensor& input_tensor_m,
    const tt::tt_fabric::Topology& topology,
    const MeshCoordinate& root_coord,
    float scale_fp32,
    const std::optional<Tensor>& optional_output_tensor_l,
    const std::optional<Tensor>& optional_output_tensor_s,
    const std::optional<Tensor>& optional_output_tensor_m,
    const std::optional<Tensor>& optional_intermediate_tensor,
    const std::optional<std::vector<ttnn::CoreCoord>>& input_mux_cores) {
    using OperationType = ttnn::operations::ccl::ReduceToRootOp;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            root_coord,
            scale_fp32,
            topology,
            input_mux_cores,
            {input_tensor_l.tensor_spec(), input_tensor_s.tensor_spec(), input_tensor_m.tensor_spec()}},
        OperationType::tensor_args_t{
            input_tensor_l,
            input_tensor_s,
            input_tensor_m,
            optional_output_tensor_l,
            optional_output_tensor_s,
            optional_output_tensor_m,
            optional_intermediate_tensor});
}
}  // namespace ttnn::prim
