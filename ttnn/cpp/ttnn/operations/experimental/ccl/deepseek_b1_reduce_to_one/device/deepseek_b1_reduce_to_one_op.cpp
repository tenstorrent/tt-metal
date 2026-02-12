// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

#include "deepseek_b1_reduce_to_one_op.hpp"

using namespace tt::tt_metal;
namespace ttnn::operations::experimental::ccl {

void DeepseekB1ReduceToOneOp::validate(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;

    auto* mesh_device = input.device();

    // Validate mesh shape: must be exactly 4 rows x 2 columns
    const auto mesh_shape = mesh_device->shape();
    TT_FATAL(
        mesh_shape.dims() == 2 && mesh_shape[0] == 4 && mesh_shape[1] == 2,
        "Mesh shape must be exactly 4x2 (4 rows x 2 columns), got {}x{}",
        mesh_shape[0],
        mesh_shape[1]);

    // Validate root_coord: must be in middle rows (1 or 2)
    const auto& root_coord = operation_attributes.root_coord;
    TT_FATAL(
        root_coord[0] == 1 || root_coord[0] == 2,
        "Root coordinate row must be 1 or 2 (middle rows), got {}",
        root_coord[0]);
    TT_FATAL(root_coord[1] < 2, "Root coordinate column must be 0 or 1, got {}", root_coord[1]);

    // Validate exit_coord: must be within mesh bounds
    const auto& exit_coord = operation_attributes.exit_coord;
    TT_FATAL(exit_coord[0] < 4, "Exit coordinate row must be less than 4, got {}", exit_coord[0]);
    TT_FATAL(exit_coord[1] < 2, "Exit coordinate column must be less than 2, got {}", exit_coord[1]);

    const auto& optional_output_tensor = tensor_args.optional_output_tensor;
    if (optional_output_tensor.has_value()) {
        TT_FATAL(
            optional_output_tensor.value().device() == mesh_device,
            "Output tensor must be allocated on same mesh device as input tensor");

        // Output tensor can have different sharding (e.g., single-core for gather)
        // but must have same logical shape and dtype
        const auto& output_spec = optional_output_tensor.value().tensor_spec();
        const auto& input_spec = input.tensor_spec();
        TT_FATAL(
            output_spec.logical_shape() == input_spec.logical_shape(),
            "Output tensor logical shape {} does not match input logical shape {}",
            output_spec.logical_shape(),
            input_spec.logical_shape());
        TT_FATAL(
            output_spec.tensor_layout().get_data_type() == input_spec.tensor_layout().get_data_type(),
            "Output tensor dtype does not match input dtype");
        TT_FATAL(optional_output_tensor.value().is_sharded(), "Output tensor must be sharded");

        // Output tensor must be sharded on exactly 1 core for gather
        const auto& output_shard_spec = optional_output_tensor.value().shard_spec().value();
        uint32_t num_output_cores = 0;
        for (const auto& core_range : output_shard_spec.grid.ranges()) {
            num_output_cores += core_range.size();
        }
        TT_FATAL(num_output_cores == 1, "Output tensor must be sharded on exactly 1 core, got {}", num_output_cores);
    }

    const auto& optional_intermediate_tensors = tensor_args.optional_intermediate_tensors;
    if (optional_intermediate_tensors.has_value()) {
        TT_FATAL(
            optional_intermediate_tensors.value().size() == 3,
            "Expected 3 intermediate tensors for 3 reduction rounds, got {}",
            optional_intermediate_tensors.value().size());

        const auto& input_spec = input.tensor_spec();
        for (size_t i = 0; i < optional_intermediate_tensors.value().size(); i++) {
            const auto& tensor = optional_intermediate_tensors.value()[i];
            TT_FATAL(
                tensor.device() == mesh_device,
                "Intermediate tensor {} must be allocated on same mesh device as input tensor",
                i);
            TT_FATAL(tensor.is_sharded(), "Intermediate tensor {} must be sharded", i);

            // Each intermediate buffer receives one complete shard, so must match input spec
            const auto& tensor_spec = tensor.tensor_spec();
            TT_FATAL(
                tensor_spec.logical_shape() == input_spec.logical_shape(),
                "Intermediate tensor {} logical shape {} does not match input logical shape {}",
                i,
                tensor_spec.logical_shape(),
                input_spec.logical_shape());
            TT_FATAL(
                tensor_spec.tensor_layout().get_data_type() == input_spec.tensor_layout().get_data_type(),
                "Intermediate tensor {} dtype does not match input dtype",
                i);
        }
    }

    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const uint32_t input_page_size_bytes = input.tensor_spec().compute_page_size_bytes();

    TT_FATAL(
        input_page_size_bytes % l1_alignment == 0 || input_page_size_bytes == l1_alignment,
        "Tensor page size must be aligned");

    TT_FATAL(input.is_sharded(), "Input tensor must be sharded");
};

DeepseekB1ReduceToOneOp::spec_return_value_t DeepseekB1ReduceToOneOp::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto* mesh_device = input_tensor.device();

    // 3 intermediate tensors for 3 rounds of receiving (prevents data overwrites)
    // - intermediate_r1: LEAF → ROOT* (round 1)
    // - intermediate_r2: ROOT3 → ROOT2/ROOT1 (round 2)
    // - intermediate_r3: ROOT2 → ROOT1 (round 3)
    // All buffers have the same size as the input tensor because each sender transmits
    // its complete shard in every round (no partial sends or aggregated multi-shard sends).
    std::vector<TensorSpec> intermediate_specs = {
        input_tensor.tensor_spec(), input_tensor.tensor_spec(), input_tensor.tensor_spec()};

    // Output tensor: sharded on single core (bottom-right of compute grid)
    // All 8 shards from input are gathered to this single core
    auto compute_grid = mesh_device->compute_with_storage_grid_size();
    CoreCoord output_core = {compute_grid.x - 1, compute_grid.y - 1};
    CoreRangeSet output_grid = CoreRangeSet({CoreRange(output_core, output_core)});

    const auto& input_spec = input_tensor.tensor_spec();
    const auto& input_shard_spec = input_tensor.shard_spec().value();

    // Output shard shape = full tensor (all shards combined)
    // Input is width-sharded across N cores, output is full width on 1 core
    auto input_shape = input_spec.logical_shape();
    std::array<uint32_t, 2> output_shard_shape = {input_shape[-2], input_shape[-1]};

    ShardSpec output_shard_spec(output_grid, output_shard_shape, input_shard_spec.orientation);
    MemoryConfig output_mem_config(TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1, output_shard_spec);

    TensorSpec output_spec(
        input_spec.logical_shape(),
        TensorLayout(input_spec.data_type(), PageConfig(input_spec.layout(), input_spec.tile()), output_mem_config));

    std::vector<TensorSpec> final_output_specs = {output_spec};

    return {intermediate_specs, final_output_specs};
}

DeepseekB1ReduceToOneOp::tensor_return_value_t DeepseekB1ReduceToOneOp::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(operation_attributes, tensor_args);

    auto* mesh_device = tensor_args.input_tensor.device();

    std::vector<ttnn::Tensor> intermediate_tensors;
    std::vector<ttnn::Tensor> final_output_tensors;

    // Use provided intermediate tensors or create fresh ones
    // For trace mode to work, we must reuse the same tensors passed from host
    if (tensor_args.optional_intermediate_tensors.has_value()) {
        // Use all 3 provided intermediate tensors
        for (const auto& tensor : tensor_args.optional_intermediate_tensors.value()) {
            intermediate_tensors.push_back(tensor);
        }
    } else {
        // Create 3 fresh intermediate tensors for 3 rounds of receiving
        for (size_t i = 0; i < 3; ++i) {
            intermediate_tensors.push_back(create_device_tensor(output_specs[0][i], mesh_device));
        }
    }

    // Create or use provided output tensor
    if (tensor_args.optional_output_tensor.has_value()) {
        final_output_tensors.push_back(tensor_args.optional_output_tensor.value());
    } else {
        final_output_tensors.push_back(create_device_tensor(output_specs[1][0], mesh_device));
    }

    return {intermediate_tensors, final_output_tensors};
}

DeepseekB1ReduceToOneOp::DeepseekB1ReduceToOne::cached_mesh_workload_t
DeepseekB1ReduceToOneOp::DeepseekB1ReduceToOne::create_mesh_workload(
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
    semaphores.reserve(4);
    // 4 semaphores: round1, round2, round3 (reduction tree), exit (ROOT1 to exit_coord)
    for (size_t i = 0; i < 4; ++i) {
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

device_operation::CachedProgram<DeepseekB1ReduceToOneOp::DeepseekB1ReduceToOne::shared_variables_t>
DeepseekB1ReduceToOneOp::DeepseekB1ReduceToOne::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    std::optional<MeshCoordinate>& forward_coord,
    std::optional<MeshCoordinate>& backward_coord,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    std::vector<tt::tt_metal::GlobalSemaphore>& semaphores) {
    const auto& root_coordinate = operation_attributes.root_coord;
    const auto& exit_coordinate = operation_attributes.exit_coord;

    return deepseek_b1_reduce_to_one_program_factory(
        tensor_args,
        operation_attributes,
        root_coordinate,
        exit_coordinate,
        mesh_coordinate,
        forward_coord,
        backward_coord,
        tensor_return_value,
        semaphores);
}

}  // namespace ttnn::operations::experimental::ccl

namespace ttnn::prim {
ttnn::operations::experimental::ccl::DeepseekB1ReduceToOneOp::tensor_return_value_t deepseek_b1_reduce_to_one(
    const Tensor& input_tensor,
    const tt::tt_fabric::Topology& topology,
    const MeshCoordinate& root_coord,
    const MeshCoordinate& exit_coord,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<std::vector<Tensor>>& optional_intermediate_tensors) {
    using OperationType = ttnn::operations::experimental::ccl::DeepseekB1ReduceToOneOp;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{root_coord, exit_coord, topology, input_tensor.tensor_spec()},
        OperationType::tensor_args_t{input_tensor, optional_output_tensor, optional_intermediate_tensors});
}
}  // namespace ttnn::prim
