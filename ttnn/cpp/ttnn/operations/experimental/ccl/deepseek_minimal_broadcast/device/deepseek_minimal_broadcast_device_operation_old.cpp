// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_minimal_broadcast_op.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn {

void DeepseekMinimalBroadcast::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Error, Input tensor size should be 1 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to broadcast need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to broadcast need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);
    TT_FATAL(
        this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout());

    // input tensor dtype should be bfloat16
    TT_FATAL(input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16, "Input tensor must be bfloat16");
    // input shard grid should be a single core
    const auto& shard_spec = input_tensor.shard_spec().value();
    const auto& shard_grid = shard_spec.grid;
    std::vector<CoreCoord> cores;
    for (const auto& core_range : shard_grid.ranges()) {
        auto c = corerange_to_cores(core_range, std::nullopt);
        cores.insert(cores.end(), c.begin(), c.end());
    }
    TT_FATAL(cores.size() == 1, "Input tensor must be sharded to a single core");

    // input should be tiny tile (1,32)
    const auto tile_width = input_tensor.tensor_spec().tile().get_width();
    const auto tile_height = input_tensor.tensor_spec().tile().get_height();
    TT_FATAL(input_tensor.layout() == ttnn::TILE_LAYOUT, "Input tensor must be in TILE_LAYOUT");
    TT_FATAL(
        tile_width == 32 && tile_height == 1,
        "Input tensor must be in tile size (1,32). Got tile size: ({}, {})",
        tile_height,
        tile_width);
    // input shape should be (1,1536)
    // To do add shape (1,7168) once fabric supports larger packets
    const auto& input_shape = input_tensor.logical_shape();
    TT_FATAL(
        input_shape[0] == 1 && input_shape[1] == 1536,
        "Input tensor shape must be (1,1536). Got shape: ({}, {})",
        input_shape[0],
        input_shape[1]);

    // validate same configuration for output tensors
    for (const auto& output_tensor_opt : output_tensors) {
        TT_FATAL(output_tensor_opt.has_value(), "Output tensor is not allocated");
        const auto& output_tensor = output_tensor_opt.value();
        TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Operands to broadcast need to be on device!");
        TT_FATAL(output_tensor.buffer() != nullptr, "Operands to broadcast need to be allocated in buffers on device!");
        TT_FATAL(output_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16, "Output tensor must be bfloat16");
        const auto& out_shard_spec = output_tensor.shard_spec().value();
        const auto& out_shard_grid = out_shard_spec.grid;
        std::vector<CoreCoord> out_cores;
        for (const auto& core_range : out_shard_grid.ranges()) {
            auto c = corerange_to_cores(core_range, std::nullopt);
            out_cores.insert(out_cores.end(), c.begin(), c.end());
        }
        TT_FATAL(out_cores.size() == 1, "Output tensor must be sharded to a single core");
    }
}

std::vector<ttnn::TensorSpec> DeepseekMinimalBroadcast::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    const auto& shape = input_tensor.logical_shape();
    std::vector<TensorSpec> output_specs;
    output_specs.reserve(this->ring_size);
    for (uint32_t i = 0; i < this->ring_size; ++i) {
        output_specs.push_back(TensorSpec(
            shape, TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_mem_config)));
    }
    return output_specs;
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks DeepseekMinimalBroadcast::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto* mesh_device = input_tensors[0].device();
    auto sub_device_id = this->sub_device_id;

    auto subdevice = sub_device_id.has_value() ? *sub_device_id : mesh_device->get_sub_device_ids().at(0);
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice);
    auto subdevices = {subdevice};

    // input tensor is on a single device
    auto init_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    auto final_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");

    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(
                coord, input_tensors, output_tensors, init_barrier_semaphore, final_barrier_semaphore);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks DeepseekMinimalBroadcast::create_program_at(
    const MeshCoordinate& coord,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors,
    const GlobalSemaphore& init_barrier_semaphore,
    const GlobalSemaphore& final_barrier_semaphore) const {
    log_debug(tt::LogOp, "DEBUG: create_program_at physical coordinate {} is called", coord);

    const auto& input_tensor = input_tensors[0];
    auto output_tensor = output_tensors[0];

    uint32_t target_ring_size = ccl::get_topological_dimension(input_tensor, this->cluster_axis);

    uint32_t device_index = ccl::get_linearized_index_from_physical_coord(input_tensor, coord, this->cluster_axis);

    std::optional<MeshCoordinate> forward_coord =
        ccl::get_physical_neighbor_from_physical_coord(input_tensor, coord, 1, this->topology, this->cluster_axis);

    std::optional<MeshCoordinate> backward_coord =
        ccl::get_physical_neighbor_from_physical_coord(input_tensor, coord, -1, this->topology, this->cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "DEBUG: forward_coord or backward_coord is null");

    return broadcast_batch1(
        input_tensor,
        this->sender_coord,
        coord,
        forward_coord,
        backward_coord,
        output_tensor,
        this->num_links,
        target_ring_size,
        device_index,
        this->topology,
        final_barrier_semaphore,
        init_barrier_semaphore,
        this->sub_device_id);
}

tt::tt_metal::operation::Hash DeepseekMinimalBroadcast::compute_program_hash(
    const std::vector<Tensor>& input_tensors) const {
    log_trace(tt::LogOp, "compute_program_hash is called");
    auto input_shape = input_tensors[0].padded_shape();
    auto input_memory_layout = input_tensors[0].layout();
    auto input_dtype = input_tensors[0].dtype();
    auto input_memory_config = input_tensors[0].memory_config();
    return tt::tt_metal::operation::hash_operation<DeepseekMinimalBroadcast>(
        this->sender_coord,
        this->num_links,
        this->ring_size,
        this->output_mem_config,
        this->topology,
        this->cluster_axis,
        this->sub_device_id.has_value(),
        this->sub_device_id.has_value()
            ? input_tensors[0].device()->worker_cores(
                  tt::tt_metal::HalProgrammableCoreType::TENSIX, this->sub_device_id.value())
            : CoreRangeSet(CoreRange({0, 0}, {0, 0})),
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

namespace operations::experimental::ccl {

Tensor deepseek_minimal_broadcast_impl(
    const Tensor& input_tensor,
    const MeshCoordinate& sender_coord,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    const auto& tensor_topology = input_tensor.tensor_topology();
    const auto& tensor_topology_shape = tensor_topology.distribution_shape();

    if (!cluster_axis.has_value()) {
        TT_FATAL(
            tensor_topology_shape.is_line_topology(),
            "minimal deepseek broadcast op is only supported for a linear tensor topology shape");
    }

    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);

    TT_FATAL(
        num_devices > 1,
        "broadcast op will only work for num_devices > 1, but has {}, shape: {}",
        num_devices,
        tensor_topology_shape);

    ttnn::ccl::Topology ccl_topology = topology;
    TT_FATAL(
        ccl_topology == ttnn::ccl::Topology::Linear,
        "Currently only Linear topology is supported in deepseek minimal broadcast op, but got {}",
        ccl_topology);

    auto outputs = tt::tt_metal::operation::run(
        ttnn::DeepseekMinimalBroadcast(
            sender_coord,
            num_links,
            num_devices,
            memory_config.value_or(input_tensor.memory_config()),
            ccl_topology,
            sub_device_id,
            cluster_axis),
        {input_tensor});
    return outputs.at(0);
}

Tensor deepseek_minimal_broadcast(
    const Tensor& input_tensor,
    const MeshCoordinate& sender_coord,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    return deepseek_minimal_broadcast_impl(
        input_tensor, sender_coord, num_links, memory_config, topology, cluster_axis, sub_device_id);
}

}  // namespace operations::experimental::ccl

}  // namespace ttnn
