// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_broadcast_op.hpp"
#include "fused_broadcast_program.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn {

void FusedBroadcast::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Error, Input tensor size should be 1 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to fused_broadcast need to be on device!");
    TT_FATAL(
        input_tensor.buffer() != nullptr, "Operands to fused_broadcast need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);

    // Validate mesh dimensions
    TT_FATAL(mesh_shape[0] > 0 && mesh_shape[1] > 0, "Invalid mesh shape");
    TT_FATAL(root_coord[0] < mesh_shape[0] && root_coord[1] < mesh_shape[1], "Root coordinate out of bounds");

    // Validate for 4x2 mesh (current optimization target)
    // TT_FATAL(mesh_shape[0] == 4 && mesh_shape[1] == 2, "FusedBroadcast currently optimized for 4x2 mesh");
    // TT_FATAL(root_coord[0] == 1 || root_coord[0] == 2, "Root should be in compute row (1 or 2) for optimal latency");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout());
}

std::vector<ttnn::TensorSpec> FusedBroadcast::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    const auto& shape = input_tensor.logical_shape();

    // Single output tensor on current device
    std::vector<TensorSpec> output_specs;
    output_specs.push_back(TensorSpec(
        shape, TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_mem_config)));

    return output_specs;
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks FusedBroadcast::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto mesh_device = input_tensors[0].device();
    auto sub_device_id = this->sub_device_id;
    auto subdevice = sub_device_id.has_value() ? *sub_device_id : mesh_device->get_sub_device_ids().at(0);
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice);
    auto subdevices = {subdevice};

    auto coordination_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    auto barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);

    return ttnn::ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors, coordination_semaphore, barrier_semaphore);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks FusedBroadcast::create_program_at(
    const MeshCoordinate& coord,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors,
    const GlobalSemaphore& coordination_semaphore,
    const GlobalSemaphore& barrier_semaphore) const {
    // Find the correct input/output tensor for this device
    const auto& input_tensor = input_tensors[0];
    auto& output_tensor = output_tensors[0];
    auto device = input_tensor.device();
    auto mesh_device = device->get_mesh_device();
    auto device_coord = coord;
    uint32_t ring_index = device_coord[0] * mesh_shape[1] + device_coord[1];
    return fused_broadcast_multicore(
        input_tensor,
        output_tensor,
        device_coord,
        root_coord,
        mesh_shape,
        num_links,
        ring_size,
        ring_index,
        topology,
        coordination_semaphore,
        barrier_semaphore,
        sub_device_id);
}

tt::tt_metal::operation::Hash FusedBroadcast::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];

    return tt::tt_metal::operation::hash_operation<FusedBroadcast>(
        root_coord,
        mesh_shape,
        num_links,
        ring_size,
        topology,
        input_tensor.dtype(),
        input_tensor.memory_config(),
        input_tensor.logical_shape(),
        input_tensor.layout());
}

namespace operations::ccl {

ttnn::Tensor fused_broadcast_impl(
    const Tensor& input_tensor,
    const MeshCoordinate& root_coord,
    const MeshCoordinate& mesh_shape,
    uint32_t num_links,
    uint32_t ring_size,
    Topology topology,
    const MemoryConfig& output_mem_config,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    auto fused_broadcast_op =
        FusedBroadcast{root_coord, mesh_shape, num_links, ring_size, output_mem_config, topology, sub_device_id};

    return tt::tt_metal::operation::run(fused_broadcast_op, {input_tensor}).at(0);
}

}  // namespace operations::ccl

}  // namespace ttnn
