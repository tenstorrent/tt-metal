// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "persistent_moe_request_device_operation.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "d2d_routes.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

namespace {

using tt::tt_metal::distributed::MeshSocket;
using tt::tt_metal::distributed::SocketEndpoint;

constexpr uint32_t kRequestHeaderBytes = 64;
constexpr uint32_t kRequestStopLayer = 0xFFFFFFFFu;

// This chip's side of the request socket: the core that sends the request, and the request's receivers.
struct RequestEndpoints {
    tt::tt_metal::CoreCoord core;
    std::vector<tt::tt_fabric::FabricNodeId> receivers;
};

std::optional<RequestEndpoints> request_endpoints(
    const PersistentMoeRequestParams& attributes, const ttnn::MeshCoordinate& coord) {
    std::optional<tt::tt_metal::CoreCoord> core;
    std::optional<tt::tt_metal::CoreCoord> receiver_core;
    std::vector<tt::tt_fabric::FabricNodeId> receivers;
    for (const auto& c : attributes.request_socket.get_config().socket_connection_config) {
        if (c.sender_core.device_coord != coord) {
            continue;
        }
        TT_FATAL(
            (!core.has_value() || *core == c.sender_core.core_coord) &&
                (!receiver_core.has_value() || *receiver_core == c.receiver_core.core_coord),
            "persistent_moe_request: {}'s request connections must share one sender core and one receiver core, "
            "so that the page is one multicast",
            coord);
        core = c.sender_core.core_coord;
        receiver_core = c.receiver_core.core_coord;
        receivers.push_back(
            attributes.request_socket.get_fabric_node_id(SocketEndpoint::RECEIVER, c.receiver_core.device_coord));
    }
    if (!core.has_value()) {
        return std::nullopt;
    }
    return RequestEndpoints{.core = *core, .receivers = std::move(receivers)};
}

// The partials buffer's layout (see PersistentMoeRequestParams): its sum cores, one per part of the row, the
// part's bytes, and the slots (the persistent op's chips) per shard.
struct PartialsLayout {
    std::vector<tt::tt_metal::CoreCoord> sum_cores;
    uint32_t part_bytes;
    uint32_t num_slots;
};

// The sum views a part as tiles of this shape; the sum is elementwise, so any whole-tile view does.
constexpr uint32_t kSumTileH = 16;
constexpr uint32_t kSumTileW = 32;
constexpr uint32_t kSumTileBytes = kSumTileH * kSumTileW * 2;

PartialsLayout partials_layout(const PersistentMoeRequestParams& attributes, uint32_t row_bytes) {
    const auto& partials = attributes.partials;
    const auto& shard = partials.shard_spec().value();
    PartialsLayout layout{
        .sum_cores = corerange_to_cores(shard.grid, std::nullopt, /*row_wise=*/true), .part_bytes = 0, .num_slots = 0};
    const auto parts = static_cast<uint32_t>(layout.sum_cores.size());
    TT_FATAL(
        parts > 0 && row_bytes % parts == 0 && (row_bytes / parts) % kSumTileBytes == 0,
        "persistent_moe_request: a {} B row does not split into {} parts of whole {} B tiles",
        row_bytes,
        parts,
        kSumTileBytes);
    layout.part_bytes = row_bytes / parts;
    // num_slots slots of a part, then a flag word per slot.
    layout.num_slots = static_cast<uint32_t>(partials.buffer()->aligned_size_per_bank()) / (layout.part_bytes + 4u);
    TT_FATAL(
        layout.num_slots > 0,
        "persistent_moe_request: the partials shards hold no slot of a {} B part",
        layout.part_bytes);
    return layout;
}

}  // namespace

void PersistentMoeRequestDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& input) {
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "persistent_moe_request: the input must be on device");
    TT_FATAL(
        input.layout() == Layout::ROW_MAJOR && input.dtype() == DataType::BFLOAT16 &&
            input.logical_shape().volume() == input.logical_shape()[-1] && input.buffer()->num_pages() == 1,
        "persistent_moe_request: the input must be one ROW_MAJOR bfloat16 row, got {} {} {}",
        input.logical_shape(),
        input.layout(),
        input.dtype());
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "persistent_moe_request: the input must be interleaved");
    auto* mesh = input.device();
    TT_FATAL(
        attributes.request_socket.get_socket_endpoint_type() == SocketEndpoint::SENDER &&
            attributes.request_socket.get_mesh_device() == mesh,
        "persistent_moe_request: it sends on request_socket, whose sending end must be on the input's mesh");
    const auto& partials = attributes.partials;
    TT_FATAL(
        partials.storage_type() == StorageType::DEVICE && partials.device() == mesh &&
            partials.memory_config().buffer_type() == BufferType::L1 &&
            partials.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "persistent_moe_request: partials must be an L1 HEIGHT_SHARDED tensor on the input's mesh");
    partials_layout(attributes, input.buffer()->page_size());
}

PersistentMoeRequestDeviceOperation::spec_return_value_t PersistentMoeRequestDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& input) {
    return input.tensor_spec();
}

PersistentMoeRequestDeviceOperation::tensor_return_value_t PersistentMoeRequestDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& input) {
    return create_device_tensor(compute_output_specs(attributes, input), input.device());
}

ttsl::hash::hash_t PersistentMoeRequestDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& input) {
    return tt::tt_metal::operation::hash_operation<PersistentMoeRequestDeviceOperation>(
        attributes.request_socket,
        attributes.request_socket.get_config_buffer_address(),
        attributes.partials.buffer()->address(),
        attributes.partials.memory_config(),
        input);
}

tt::tt_metal::ProgramDescriptor PersistentMoeRequestDeviceOperation::MultiCore::create_descriptor(
    const operation_attributes_t& attributes,
    const tensor_args_t& input,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    using namespace tt::tt_metal;
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "persistent_moe_request: built per mesh coordinate");
    const auto& coord = *mesh_dispatch_coordinate;
    const auto endpoints = request_endpoints(attributes, coord);
    if (!endpoints.has_value()) {
        return ProgramDescriptor{};
    }
    auto* mesh = input.device();
    const auto src = mesh->get_fabric_node_id(coord);
    const auto routes = d2d_routes(src, endpoints->receivers);
    const auto& core = endpoints->core;
    const uint32_t row_bytes = input.buffer()->page_size();
    const uint32_t page_bytes = kRequestHeaderBytes + row_bytes;
    const bool stop = attributes.layer == kRequestStopLayer;

    ProgramDescriptor desc;
    constexpr uint32_t cb_page = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = page_bytes,
        .core_ranges = CoreRangeSet(CoreRange(core, core)),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_page,
            .data_format = tt::DataFormat::UInt32,
            .page_size = page_bytes,
        }}},
    });

    std::vector<uint32_t> ct_args = {
        static_cast<uint32_t>(attributes.request_socket.get_config_buffer_address()),
        row_bytes,
        static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes()),
        cb_page,
    };
    TensorAccessorArgs(*input.buffer()).append_to(ct_args);
    KernelDescriptor kernel;
    kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/kernels/dataflow/moe_request.cpp";
    kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = CoreRangeSet(CoreRange(core, core));
    kernel.compile_time_args = std::move(ct_args);
    kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    tt::tt_metal::KernelHandle kernel_id = desc.kernels.size();
    desc.kernels.push_back(std::move(kernel));

    std::vector<uint32_t> args = {attributes.layer, static_cast<uint32_t>(input.buffer()->address())};
    TT_FATAL(
        routes.size() <= 16,
        "persistent_moe_request: {} sends to {} chips, more than the kernel's 16",
        coord,
        routes.size());
    args.push_back(static_cast<uint32_t>(routes.size()));
    for (const auto& route : routes) {
        append_d2d_route_args(route, args);
    }
    tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
        *mesh, src, {routes.front().next}, {0u}, desc, kernel_id, core, args, tt::tt_fabric::FabricApiType::Mesh);
    desc.kernels[kernel_id].runtime_args.emplace_back(
        core, KernelDescriptor::CoreRuntimeArgs(args.begin(), args.end()));

    // The sum of each part of the row, on its sum core: the slots are a CB over the partials shard, which the
    // persistent op's chips fill.
    const auto layout = partials_layout(attributes, row_bytes);
    const uint32_t tiles_per_part = layout.part_bytes / kSumTileBytes;
    CoreRangeSet sum_set;
    for (const auto& c : layout.sum_cores) {
        sum_set = sum_set.merge(CoreRangeSet(CoreRange(c, c)));
    }
    const TileDescriptor sum_tile(tt::tt_metal::Tile({kSumTileH, kSumTileW}));
    constexpr uint32_t cb_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = layout.num_slots * layout.part_bytes,
        .core_ranges = sum_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_in,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = kSumTileBytes,
            .tile = sum_tile,
        }}},
        .buffer = attributes.partials.buffer(),
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = layout.part_bytes,
        .core_ranges = sum_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_out,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = kSumTileBytes,
            .tile = sum_tile,
        }}},
    });
    std::vector<uint32_t> sum_ct_args = {layout.num_slots, layout.part_bytes, cb_in, cb_out, tiles_per_part};
    TensorAccessorArgs(*output.buffer()).append_to(sum_ct_args);
    KernelDescriptor sum_reader;
    sum_reader.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/kernels/dataflow/moe_partials_sum.cpp";
    sum_reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sum_reader.core_ranges = sum_set;
    sum_reader.compile_time_args = std::move(sum_ct_args);
    // RISCV_1: the request core may be a sum core, and the request kernel is RISCV_0.
    sum_reader.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};
    KernelDescriptor sum_compute;
    sum_compute.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/kernels/compute/moe_partials_sum.cpp";
    sum_compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sum_compute.core_ranges = sum_set;
    sum_compute.compile_time_args = {layout.num_slots, tiles_per_part, cb_in, cb_out};
    sum_compute.config = ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false};
    for (uint32_t part = 0; part < layout.sum_cores.size(); ++part) {
        const auto& c = layout.sum_cores[part];
        sum_reader.emplace_runtime_args(c, {stop ? 1u : 0u, attributes.partials.buffer(), output.buffer(), part});
        sum_compute.emplace_runtime_args(c, {stop ? 1u : 0u});
    }
    desc.kernels.push_back(std::move(sum_reader));
    desc.kernels.push_back(std::move(sum_compute));
    return desc;
}

void PersistentMoeRequestDeviceOperation::MultiCore::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& attributes,
    const tensor_args_t& input,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "persistent_moe_request: built per mesh coordinate");
    const auto endpoints = request_endpoints(attributes, *mesh_dispatch_coordinate);
    if (!endpoints.has_value()) {
        return;
    }
    const uint32_t skip = attributes.layer == kRequestStopLayer ? 1u : 0u;
    auto& args = tt::tt_metal::GetRuntimeArgs(program, 0, endpoints->core);
    args[0] = attributes.layer;
    args[1] = input.buffer()->address();
    // Kernels 1 and 2: the sums' data movement and compute, on every sum core.
    const auto layout = partials_layout(attributes, input.buffer()->page_size());
    for (const auto& c : layout.sum_cores) {
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, 1, c);
        reader_args[0] = skip;
        reader_args[2] = output.buffer()->address();
        tt::tt_metal::GetRuntimeArgs(program, 2, c)[0] = skip;
    }
}

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts

namespace ttnn::prim {

Tensor persistent_moe_request(
    const Tensor& input_tensor,
    uint32_t layer,
    const tt::tt_metal::distributed::MeshSocket& request_socket,
    const Tensor& partials) {
    using OperationType =
        ttnn::operations::experimental::deepseek::moe::fused_experts::PersistentMoeRequestDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t(layer, request_socket, partials), input_tensor);
}

}  // namespace ttnn::prim
