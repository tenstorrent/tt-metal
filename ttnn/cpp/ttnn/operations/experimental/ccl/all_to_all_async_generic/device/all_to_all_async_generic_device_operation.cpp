// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_generic_device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>

namespace {

struct DrainCoreMapping {
    std::vector<ttnn::CoreCoord> logical_core_candidates;
    std::vector<ttnn::CoreCoord> virtual_cores;
};

DrainCoreMapping gather_drain_virtual_cores(
    const ttnn::Tensor& input_tensor, const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    struct DrainCoreRecord {
        uint32_t valid = 0;
        uint32_t x = 0;
        uint32_t y = 0;
    };

    auto* mesh_device = input_tensor.device();
    const auto subdevice = sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const auto available_core_count =
        mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice).num_cores();
    TT_FATAL(available_core_count > 0, "All-to-all cannot select its drain synchronization core");
    // Direct schedules drain on selected core 0. Mux schedules place the mux on core 0 and drain on sender core 1.
    // Use the same allocation helper as the program factory so non-rectangular subdevices produce the same prefix.
    auto [candidate_core_range, logical_core_candidates] =
        ttnn::ccl::choose_worker_cores(1, std::min<size_t>(2, available_core_count), mesh_device, sub_device_id);
    (void)candidate_core_range;
    const auto mesh_shape = mesh_device->shape();
    const size_t candidates_per_node = logical_core_candidates.size();

    std::vector<DrainCoreRecord> local_records(mesh_shape.mesh_size() * candidates_per_node);
    auto maybe_device_it = mesh_device->get_view().begin();
    for (const auto& coord : ttnn::MeshCoordinateRange(mesh_shape)) {
        const auto& maybe_coordinate_device = *maybe_device_it++;
        if (maybe_coordinate_device.is_remote()) {
            continue;
        }
        const size_t node_index = coord.to_linear_index(mesh_shape);
        for (size_t candidate = 0; candidate < candidates_per_node; ++candidate) {
            const auto drain_virtual_core =
                maybe_coordinate_device.value()->worker_core_from_logical_core(logical_core_candidates[candidate]);
            local_records[node_index * candidates_per_node + candidate] = {
                .valid = 1,
                .x = static_cast<uint32_t>(drain_virtual_core.x),
                .y = static_cast<uint32_t>(drain_virtual_core.y)};
        }
    }

    // Like the device collective itself, this host exchange requires SPMD invocation by every mesh rank. Every rank
    // enters it on every Fabric2D invocation, independently of its local program-cache state. Do not process-cache
    // this exchange: asymmetric caches across ranks would let some ranks skip the collective and deadlock the others.
    // Exchanging only the two possible drain coordinates per mesh node keeps heterogeneous harvesting correct without
    // putting a collective in the program factory or exposing remote device translation through MeshDevice.
    const auto distributed_context = tt::tt_metal::DistributedHostBuffer::create(mesh_device->get_view()).context();
    const size_t world_size = *distributed_context->size();
    std::vector<DrainCoreRecord> gathered_records(local_records.size() * world_size);
    if (world_size == 1) {
        gathered_records = local_records;
    } else {
        distributed_context->all_gather(
            ttsl::as_writable_bytes(ttsl::Span<DrainCoreRecord>{local_records}),
            ttsl::as_writable_bytes(ttsl::Span<DrainCoreRecord>{gathered_records}));
    }

    std::vector<ttnn::CoreCoord> drain_virtual_cores(local_records.size());
    std::vector<uint32_t> owner_counts(local_records.size(), 0);
    for (size_t rank = 0; rank < world_size; ++rank) {
        for (size_t node = 0; node < local_records.size(); ++node) {
            const auto& record = gathered_records[rank * local_records.size() + node];
            if (record.valid == 0) {
                continue;
            }
            TT_FATAL(
                owner_counts[node] == 0,
                "All-to-all mesh node {} drain candidate {} is owned by more than one distributed rank",
                node / candidates_per_node,
                node % candidates_per_node);
            drain_virtual_cores[node] = {record.x, record.y};
            owner_counts[node] = 1;
        }
    }
    for (size_t node = 0; node < owner_counts.size(); ++node) {
        TT_FATAL(
            owner_counts[node] == 1,
            "No distributed rank owns all-to-all mesh node {} drain candidate {}",
            node / candidates_per_node,
            node % candidates_per_node);
    }
    return {
        .logical_core_candidates = std::move(logical_core_candidates), .virtual_cores = std::move(drain_virtual_cores)};
}

}  // namespace

namespace ttnn::experimental::prim {

void AllToAllAsyncGenericDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    const auto& input_tensor = tensor_args.input_tensor;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_to_all_async must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_to_all_async must be allocated in buffers on device");

    const auto& page_size = input_tensor.buffer()->page_size();
    const auto& input_shape = input_tensor.logical_shape();
    auto rank = input_shape.rank();
    auto* mesh_device = input_tensor.device();
    const auto subdevice_id = operation_attributes.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const auto available_worker_cores =
        mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id).num_cores();
    const auto max_payload_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();

    TT_FATAL(operation_attributes.in_dim >= 0 && operation_attributes.in_dim < rank, "in_dim out of range");
    TT_FATAL(operation_attributes.out_dim >= 0 && operation_attributes.out_dim < rank, "out_dim out of range");

    TT_FATAL(page_size % input_tensor.buffer()->alignment() == 0, "AllToAllAsync currently requires aligned pages");

    TT_FATAL(
        available_worker_cores >= operation_attributes.num_links,
        "All-to-all requires at least one worker per link: requested {} links, but subdevice has {} workers",
        operation_attributes.num_links,
        available_worker_cores);
    TT_FATAL(
        max_payload_size >= page_size,
        "Fabric maximum payload {} must fit at least one tensor page of size {}",
        max_payload_size,
        page_size);

    TT_FATAL(
        input_shape[operation_attributes.out_dim] % operation_attributes.num_devices == 0,
        "AllToAllAsync: input tensor dimension {} must be divisible by num_devices {}",
        input_shape[operation_attributes.out_dim],
        operation_attributes.num_devices);
    TT_FATAL(input_tensor.layout() == Layout::TILE, "Unsupported input layout {}.", input_tensor.layout());

    // recreate output shape to cover optional output buffer
    auto output_shape = input_tensor.logical_shape();
    output_shape[operation_attributes.in_dim] *= operation_attributes.num_devices;
    output_shape[operation_attributes.out_dim] /= operation_attributes.num_devices;

    // Check padding support, currently supported only on height
    auto last_dim = rank - 1;
    auto second_last_dim = rank - 2;
    TT_FATAL(
        operation_attributes.in_dim != second_last_dim || input_shape[operation_attributes.in_dim] % 16 == 0,
        "{} dimension support only 0 or 16 padding, so must be divisible by 16. Input tensor shape {} , but has {} "
        "padding",
        operation_attributes.in_dim,
        input_shape,
        input_shape[operation_attributes.in_dim] % 32);
    TT_FATAL(
        operation_attributes.out_dim != second_last_dim || output_shape[operation_attributes.out_dim] % 16 == 0,
        "{} dimension support only 0 or 16 padding, so must be divisible by 16. Output tensor shape {} , but has {} "
        "padding",
        operation_attributes.out_dim,
        output_shape,
        output_shape[operation_attributes.out_dim] % 32);
    TT_FATAL(
        operation_attributes.in_dim != last_dim || input_shape[operation_attributes.in_dim] % 32 == 0,
        "{} dimension doesn't support padding, so must be divisible by 32. Input tensor shape {} , but has {} padding",
        operation_attributes.in_dim,
        input_shape,
        input_shape[operation_attributes.in_dim] % 32);
    TT_FATAL(
        operation_attributes.out_dim != last_dim || output_shape[operation_attributes.out_dim] % 32 == 0,
        "{} dimension doesn't support padding, so must be divisible by 32. Output tensor shape {} , but has {} padding",
        operation_attributes.out_dim,
        output_shape,
        output_shape[operation_attributes.out_dim] % 32);
}

void AllToAllAsyncGenericDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig())) {
        TT_FATAL(
            operation_attributes.cluster_axis.has_value(),
            "all_to_all_async_generic on FABRIC_2D requires a cluster_axis");
    }

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& persistent_output_buffer = tensor_args.persistent_output_buffer;

    if (persistent_output_buffer.has_value()) {
        const auto& output_tensor = persistent_output_buffer.value();

        TT_FATAL(
            output_tensor.storage_type() == StorageType::DEVICE,
            "Output tensor for all_to_all_async must be on device");
        TT_FATAL(
            output_tensor.buffer()->buffer_type() == BufferType::DRAM,
            "Output tensor for all_to_all_async must be in DRAM, but is in {}",
            output_tensor.buffer()->buffer_type());
        TT_FATAL(output_tensor.layout() == Layout::TILE, "Unsupported output layout {}.", output_tensor.layout());

        TT_FATAL(output_tensor.dtype() == input_tensor.dtype(), "Output tensor dtype must match input tensor dtype");
        TT_FATAL(
            output_tensor.memory_config() == operation_attributes.output_mem_config,
            "Output tensor memory config must match specified output_mem_config");

        const auto& output_shape = output_tensor.logical_shape();
        auto expected_output_shape = input_tensor.logical_shape();
        expected_output_shape[operation_attributes.in_dim] *= operation_attributes.num_devices;
        expected_output_shape[operation_attributes.out_dim] /= operation_attributes.num_devices;
        TT_FATAL(
            output_shape == expected_output_shape,
            "Output tensor shape {} must match expected output tensor shape {} for AllToAllAsync",
            output_shape,
            expected_output_shape);
    }
}

AllToAllAsyncGenericDeviceOperation::spec_return_value_t AllToAllAsyncGenericDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.persistent_output_buffer.has_value()) {
        return tensor_args.persistent_output_buffer->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    auto shape = input_tensor.logical_shape();
    shape[operation_attributes.in_dim] *= operation_attributes.num_devices;
    shape[operation_attributes.out_dim] /= operation_attributes.num_devices;
    return tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), input_tensor.tensor_spec().page_config(), operation_attributes.output_mem_config));
}

AllToAllAsyncGenericDeviceOperation::tensor_return_value_t AllToAllAsyncGenericDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.persistent_output_buffer.has_value()) {
        return tensor_args.persistent_output_buffer.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

ttsl::hash::hash_t AllToAllAsyncGenericDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "AllToAllAsyncGenericDeviceOperation::compute_program_hash is called");

    auto subdevice_id = operation_attributes.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    const auto fabric_config = tt::tt_fabric::GetFabricConfig();
    const auto max_payload_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    // The cached program contains the fabric mux kernel and client ABI. Bump this whenever either changes.
    constexpr uint32_t fabric_mux_implementation_version = 2;
    return tt::tt_metal::operation::hash_operation<AllToAllAsyncGenericDeviceOperation>(
        operation_attributes.in_dim,
        operation_attributes.out_dim,
        operation_attributes.num_links,
        operation_attributes.num_devices,
        operation_attributes.output_mem_config,
        operation_attributes.topology,
        operation_attributes.cluster_axis,
        operation_attributes.axis_is_straight,
        subdevice_core_range_set,
        fabric_config,
        operation_attributes.axis_topology,
        max_payload_size,
        fabric_mux_implementation_version,
        operation_attributes.drain_logical_core_candidates,
        operation_attributes.drain_virtual_cores,
        tensor_args);
}

Tensor all_to_all_async_generic(
    const ttnn::Tensor& input_tensor,
    const std::optional<Tensor>& persistent_output_buffer,
    int32_t in_dim,
    int32_t out_dim,
    uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> cluster_axis) {
    using OperationType = AllToAllAsyncGenericDeviceOperation;
    uint32_t num_devices = ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    TT_FATAL(num_links > 0, "all_to_all_async requires at least one fabric link");
    TT_FATAL(
        num_devices > 1,
        "all_to_all_async is a collective operation and requires more than 1 device, but has {}",
        num_devices);

    DrainCoreMapping drain_core_mapping;
    const auto fabric_config = tt::tt_fabric::GetFabricConfig();
    const bool is_fabric_2d = tt::tt_fabric::is_2d_fabric_config(fabric_config);
    const uint32_t resolved_cluster_axis = cluster_axis.value_or(0);
    const auto axis_topology = ttnn::ccl::get_axis_topology(input_tensor, fabric_config, resolved_cluster_axis);
    const bool axis_is_straight =
        !is_fabric_2d || ttnn::ccl::is_axis_straight(*input_tensor.device(), resolved_cluster_axis);
    if (is_fabric_2d) {
        drain_core_mapping = gather_drain_virtual_cores(input_tensor, sub_device_id);
    }

    auto operation_attributes = OperationType::operation_attributes_t{
        .in_dim = static_cast<uint32_t>(in_dim),
        .out_dim = static_cast<uint32_t>(out_dim),
        .num_links = num_links,
        .num_devices = num_devices,
        .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
        .topology = topology,
        .sub_device_id = sub_device_id,
        .cluster_axis = cluster_axis,
        .axis_topology = axis_topology,
        .axis_is_straight = axis_is_straight,
        .drain_logical_core_candidates = std::move(drain_core_mapping.logical_core_candidates),
        .drain_virtual_cores = std::move(drain_core_mapping.virtual_cores)};
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input_tensor, .persistent_output_buffer = persistent_output_buffer};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor all_to_all_async_generic(
    const ttnn::Tensor& input_tensor,
    const std::optional<Tensor>& persistent_output_buffer,
    int32_t in_dim,
    int32_t out_dim,
    uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> cluster_axis) {
    return ttnn::experimental::prim::all_to_all_async_generic(
        input_tensor,
        persistent_output_buffer,
        in_dim,
        out_dim,
        num_links,
        memory_config,
        topology,
        sub_device_id,
        cluster_axis);
}

}  // namespace ttnn::prim
