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

#include "reduce_to_one_op.hpp"

using namespace tt::tt_metal;
namespace ttnn::operations::ccl {

void ReduceToOneOp::validate(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;

    auto* mesh_device = input.device();

    const auto& optional_output_tensor = tensor_args.optional_output_tensor;
    if (optional_output_tensor.has_value()) {
        const auto output_specs = compute_output_specs(operation_attributes, tensor_args);

        TT_FATAL(
            output_specs[1][0] == optional_output_tensor.value().tensor_spec(),
            "Optional output tensor spec {} does not match computed output spec {}",
            optional_output_tensor.value().tensor_spec(),
            output_specs[1][0]);

        TT_FATAL(
            optional_output_tensor.value().device() == mesh_device,
            "Output tensor must be allocated on same mesh device as input tensor");
    }

    const auto& optional_intermediate_tensor = tensor_args.optional_intermediate_tensor;
    if (optional_intermediate_tensor.has_value()) {
        TT_FATAL(
            optional_intermediate_tensor.value().device() == mesh_device,
            "Intermediate tensor must be allocated on same mesh device as input tensor");
    }

    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const uint32_t input_page_size_bytes = input.tensor_spec().compute_page_size_bytes();

    TT_FATAL(
        input_page_size_bytes % l1_alignment == 0 || input_page_size_bytes == l1_alignment,
        "Tensor page size must be aligned");

    TT_FATAL(input.is_sharded(), "Input tensor must be sharded");
};

ReduceToOneOp::spec_return_value_t ReduceToOneOp::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    // Intermediate tensor has the same spec as input (used as receive buffer)
    std::vector<TensorSpec> intermediate_specs = {input_tensor.tensor_spec()};

    // Output tensor has the same spec as input
    std::vector<TensorSpec> final_output_specs = {input_tensor.tensor_spec()};

    return {intermediate_specs, final_output_specs};
}

ReduceToOneOp::tensor_return_value_t ReduceToOneOp::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(operation_attributes, tensor_args);

    auto* mesh_device = tensor_args.input_tensor.device();

    std::vector<ttnn::Tensor> intermediate_tensors;
    std::vector<ttnn::Tensor> final_output_tensors;

    // Create or use provided intermediate tensor
    if (tensor_args.optional_intermediate_tensor.has_value()) {
        intermediate_tensors.push_back(tensor_args.optional_intermediate_tensor.value());
    } else {
        intermediate_tensors.push_back(create_device_tensor(output_specs[0][0], mesh_device));
    }

    // Create or use provided output tensor
    if (tensor_args.optional_output_tensor.has_value()) {
        final_output_tensors.push_back(tensor_args.optional_output_tensor.value());
    } else {
        final_output_tensors.push_back(create_device_tensor(output_specs[1][0], mesh_device));
    }

    return {intermediate_tensors, final_output_tensors};
}

ReduceToOneOp::ReduceToOne::cached_mesh_workload_t ReduceToOneOp::ReduceToOne::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = mesh_device->get_sub_device_ids().at(0);
    auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
    semaphores.reserve(3);
    // 3 semaphores: round1 (within column), round2 (within column), round3 (cross-column)
    for (size_t i = 0; i < 3; ++i) {
        semaphores.push_back(ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0));
    }
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});

    const auto& coords = tensor_coords.coords();
    const auto& root_coord = operation_attributes.root_coord;
    const auto& mesh_shape = mesh_device->shape();
    uint32_t root_row = root_coord[0];

    for (const auto& coord : coords) {
        std::optional<MeshCoordinate> forward_coord;
        std::optional<MeshCoordinate> backward_coord;

        // Determine role to decide routing direction
        bool is_root1 = (coord == root_coord);
        bool is_root2 = (!is_root1 && coord[0] == root_row);
        // ROOT1/ROOT2: row routing (horizontal, between columns)
        // ROOT3/LEAF: column routing (vertical, between rows)

        if (is_root1 || is_root2) {
            // Row routing (horizontal): forward = next column, backward = previous column
            if (coord[1] + 1 < mesh_shape[1]) {
                forward_coord = MeshCoordinate(coord[0], coord[1] + 1);
            }
            if (coord[1] > 0) {
                backward_coord = MeshCoordinate(coord[0], coord[1] - 1);
            }
        } else {
            // Column routing (vertical): forward = next row, backward = previous row
            if (coord[0] + 1 < mesh_shape[0]) {
                forward_coord = MeshCoordinate(coord[0] + 1, coord[1]);
            }
            if (coord[0] > 0) {
                backward_coord = MeshCoordinate(coord[0] - 1, coord[1]);
            }
        }

        auto cached_workload = create_at(
            operation_attributes, coord, forward_coord, backward_coord, tensor_args, tensor_return_value, semaphores);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_workload.program));
        shared_variables.emplace(coord, std::move(cached_workload.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

device_operation::CachedProgram<ReduceToOneOp::ReduceToOne::shared_variables_t> ReduceToOneOp::ReduceToOne::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    std::optional<MeshCoordinate>& forward_coord,
    std::optional<MeshCoordinate>& backward_coord,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    std::vector<tt::tt_metal::GlobalSemaphore>& semaphores) {
    const auto& root_coordinate = operation_attributes.root_coord;

    return reduce_to_one_program_factory(
        tensor_args,
        operation_attributes,
        root_coordinate,
        mesh_coordinate,
        forward_coord,
        backward_coord,
        tensor_return_value,
        semaphores);
}

}  // namespace ttnn::operations::ccl

namespace ttnn::prim {
ttnn::operations::ccl::ReduceToOneOp::tensor_return_value_t reduce_to_one(
    const Tensor& input_tensor,
    const tt::tt_fabric::Topology& topology,
    const MeshCoordinate& root_coord,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<Tensor>& optional_intermediate_tensor) {
    using OperationType = ttnn::operations::ccl::ReduceToOneOp;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{root_coord, topology, input_tensor.tensor_spec()},
        OperationType::tensor_args_t{input_tensor, optional_output_tensor, optional_intermediate_tensor});
}
}  // namespace ttnn::prim
