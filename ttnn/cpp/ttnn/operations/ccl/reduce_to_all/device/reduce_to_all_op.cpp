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

#include "reduce_to_all_op.hpp"

using namespace tt::tt_metal;
namespace ttnn::operations::ccl {

using cached_workload_t = device_operation::CachedProgram<ReduceToAllOp::ReduceToAll::shared_variables_t>;

void ReduceToAllOp::validate(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_l = tensor_args.input_tensor_l;
    const auto& input_ms = tensor_args.input_tensor_ms;

    auto* mesh_device = input_l.device();

    // Validate MS tensor has combined format (2 columns for max and sum)
    TT_FATAL(input_ms.device() == mesh_device, "Input MS tensor must be on same mesh device as input L tensor");

    const auto& optional_output_tensor_l = tensor_args.optional_output_tensor_l;
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
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const uint32_t input_page_size_bytes = input_l.tensor_spec().compute_page_size_bytes();

    TT_FATAL(
        input_page_size_bytes % l1_alignment == 0 || input_page_size_bytes == l1_alignment,
        "Tensor page size must be aligned");
    // input mux cores (now used for aggregator) should be at least 2
    // Aggregator needs 2 cores (1 per link), mux needed 4.
    // Accept 2-4 cores for backward compatibility.
    if (operation_attributes.input_mux_cores.has_value()) {
        TT_FATAL(
            operation_attributes.input_mux_cores.value().size() >= 2,
            "Input mux/aggregator cores size must be at least 2, got {}",
            operation_attributes.input_mux_cores.value().size());
    }

    // extra worker cores should be 8
    if (operation_attributes.extra_worker_cores.has_value()) {
        TT_FATAL(
            operation_attributes.extra_worker_cores.value().size() == 8,
            "Extra worker cores size must be 8, got {}",
            operation_attributes.extra_worker_cores.value().size());
    }
};

ReduceToAllOp::spec_return_value_t ReduceToAllOp::compute_output_specs(
    const operation_attributes_t& /* operation_attributes */, const tensor_args_t& tensor_args) {
    const auto& input_tensor_l = tensor_args.input_tensor_l;
    const auto& input_tensor_ms = tensor_args.input_tensor_ms;

    // Output: only normalized L (MS is intermediate only)
    std::vector<TensorSpec> final_output_spec = {input_tensor_l.tensor_spec()};

    std::vector<TensorSpec> intermediate_specs;
    if (tensor_args.optional_fw_intermediate_tensor.has_value() &&
        tensor_args.optional_bw_intermediate_tensor.has_value() &&
        tensor_args.optional_coord_intermediate_tensor.has_value()) {
        intermediate_specs.push_back(tensor_args.optional_fw_intermediate_tensor.value().tensor_spec());
        intermediate_specs.push_back(tensor_args.optional_bw_intermediate_tensor.value().tensor_spec());
        intermediate_specs.push_back(tensor_args.optional_coord_intermediate_tensor.value().tensor_spec());
        return {intermediate_specs, final_output_spec};
    }
    // Intermediate shape: combined L + MS payload
    uint32_t shape_0 = final_output_spec[0].memory_config().shard_spec()->shape[0];
    uint32_t shape_1 = final_output_spec[0].memory_config().shard_spec()->shape[1] +
                       input_tensor_ms.tensor_spec().memory_config().shard_spec()->shape[1];
    Shape intermediate_shape = Shape{shape_0, shape_1};
    TensorSpec intermediate_spec(intermediate_shape, final_output_spec[0].tensor_layout());
    for (auto j = 0; j < 3; j++) {
        intermediate_specs.push_back(intermediate_spec);
    }

    return {intermediate_specs, final_output_spec};
}

ReduceToAllOp::tensor_return_value_t ReduceToAllOp::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(operation_attributes, tensor_args);

    auto* mesh_device = tensor_args.input_tensor_l.device();

    std::vector<ttnn::Tensor> intermediate_output_tensors;
    std::vector<ttnn::Tensor> final_output_tensors;

    auto fw_intermediate_output_tensor = create_device_tensor(output_specs.at(0)[0], mesh_device);
    if (tensor_args.optional_fw_intermediate_tensor.has_value()) {
        fw_intermediate_output_tensor = tensor_args.optional_fw_intermediate_tensor.value();
    }
    auto bw_intermediate_output_tensor = create_device_tensor(output_specs.at(0)[1], mesh_device);
    if (tensor_args.optional_bw_intermediate_tensor.has_value()) {
        bw_intermediate_output_tensor = tensor_args.optional_bw_intermediate_tensor.value();
    }
    auto coord_intermediate_output_tensor = create_device_tensor(output_specs.at(0)[2], mesh_device);
    if (tensor_args.optional_coord_intermediate_tensor.has_value()) {
        coord_intermediate_output_tensor = tensor_args.optional_coord_intermediate_tensor.value();
    }

    // Only L is final output (normalized); MS is intermediate only
    auto final_output_tensor_l = create_device_tensor(output_specs.at(1)[0], mesh_device);
    if (tensor_args.optional_output_tensor_l.has_value()) {
        final_output_tensor_l = tensor_args.optional_output_tensor_l.value();
    }

    intermediate_output_tensors = {
        fw_intermediate_output_tensor, bw_intermediate_output_tensor, coord_intermediate_output_tensor};
    final_output_tensors = {final_output_tensor_l};

    return {intermediate_output_tensors, final_output_tensors};
}

ReduceToAllOp::ReduceToAll::cached_mesh_workload_t ReduceToAllOp::ReduceToAll::create_mesh_workload(
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
    semaphores.reserve(5);
    for (size_t i = 0; i < 5; ++i) {
        semaphores.push_back(ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0));
    }
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready in reduce_to_all op");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});
    log_debug(tt::LogOp, "Synchronize devices in reduce_to_all op done");

    const auto& coords = tensor_coords.coords();
    const auto& topology = operation_attributes.topology;
    for (const auto& coord : coords) {
        std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor_l, coord, 1, topology, std::nullopt);

        std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor_l, coord, -1, topology, std::nullopt);

        if (topology == tt::tt_fabric::Topology::Ring || topology == tt::tt_fabric::Topology::Torus) {
            if (forward_coord.has_value() == 0 || backward_coord.has_value() == 0) {
                TT_FATAL(
                    false,
                    "In ring/torus topology, all devices must have both forward and backward neighbors in "
                    "reduce_to_all op");
            }
        }
        auto cached_workload = create_at(
            operation_attributes, coord, forward_coord, backward_coord, tensor_args, tensor_return_value, semaphores);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_workload.program));
        shared_variables.emplace(coord, std::move(cached_workload.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

cached_workload_t ReduceToAllOp::ReduceToAll::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    std::optional<MeshCoordinate>& forward_coord,
    std::optional<MeshCoordinate>& backward_coord,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    std::vector<tt::tt_metal::GlobalSemaphore>& semaphores) {
    const auto& root_coordinate = operation_attributes.root_coord;
    const float scale_fp32 = operation_attributes.scale_fp32;

    // TODO: Switch back to reduce_to_all_program_factory after testing
    return reduce_to_all_simplified_program_factory(
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
ttnn::operations::ccl::ReduceToAllOp::tensor_return_value_t reduce_to_all(
    const Tensor& input_tensor_l,
    const Tensor& input_tensor_ms,  // Combined: col 0 = max, col 1 = sum
    const tt::tt_fabric::Topology& topology,
    const MeshCoordinate& root_coord,
    float scale_fp32,
    const std::optional<Tensor>& optional_output_tensor_l,
    const std::optional<Tensor>& optional_fw_intermediate_tensor,
    const std::optional<Tensor>& optional_bw_intermediate_tensor,
    const std::optional<Tensor>& optional_coord_intermediate_tensor,
    const std::optional<std::vector<ttnn::CoreCoord>>& input_mux_cores,
    const std::optional<std::vector<ttnn::CoreCoord>>& extra_worker_cores,
    const std::optional<Tensor>& optional_aggregator_scratch_tensor) {
    using OperationType = ttnn::operations::ccl::ReduceToAllOp;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            root_coord,
            scale_fp32,
            topology,
            input_mux_cores,
            extra_worker_cores,
            {input_tensor_l.tensor_spec(), input_tensor_ms.tensor_spec()}},
        OperationType::tensor_args_t{
            input_tensor_l,
            input_tensor_ms,
            optional_output_tensor_l,
            optional_fw_intermediate_tensor,
            optional_bw_intermediate_tensor,
            optional_coord_intermediate_tensor,
            optional_aggregator_scratch_tensor});
}
}  // namespace ttnn::prim
