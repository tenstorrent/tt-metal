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

// This chip's side of the two sockets: the core that sends the request and receives the response,
// the request's receivers and the response's sender.
struct RequestEndpoints {
    tt::tt_metal::CoreCoord core;
    std::vector<tt::tt_fabric::FabricNodeId> receivers;
    tt::tt_fabric::FabricNodeId responder;
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
    std::optional<tt::tt_fabric::FabricNodeId> responder;
    for (const auto& c : attributes.response_socket.get_config().socket_connection_config) {
        if (c.receiver_core.device_coord != coord) {
            continue;
        }
        TT_FATAL(
            !responder.has_value() && c.receiver_core.core_coord == *core,
            "persistent_moe_request: {} must receive the response once, on its request core {}",
            coord,
            core->str());
        responder = attributes.response_socket.get_fabric_node_id(SocketEndpoint::SENDER, c.sender_core.device_coord);
    }
    TT_FATAL(responder.has_value(), "persistent_moe_request: {} sends a request but receives no response", coord);
    return RequestEndpoints{.core = *core, .receivers = std::move(receivers), .responder = *responder};
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
            attributes.response_socket.get_socket_endpoint_type() == SocketEndpoint::RECEIVER &&
            attributes.request_socket.get_mesh_device() == mesh && attributes.response_socket.get_mesh_device() == mesh,
        "persistent_moe_request: it sends on request_socket and receives on response_socket, both endpoints on the "
        "input's mesh");
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
        attributes.response_socket,
        attributes.request_socket.get_config_buffer_address(),
        attributes.response_socket.get_config_buffer_address(),
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
        static_cast<uint32_t>(attributes.response_socket.get_config_buffer_address()),
        row_bytes,
        static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes()),
        cb_page,
        kRequestStopLayer,
    };
    TensorAccessorArgs(*input.buffer()).append_to(ct_args);
    TensorAccessorArgs(*output.buffer()).append_to(ct_args);
    KernelDescriptor kernel;
    kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/kernels/dataflow/moe_request.cpp";
    kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = CoreRangeSet(CoreRange(core, core));
    kernel.compile_time_args = std::move(ct_args);
    kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    tt::tt_metal::KernelHandle kernel_id = desc.kernels.size();
    desc.kernels.push_back(std::move(kernel));

    std::vector<uint32_t> args = {
        attributes.layer,
        static_cast<uint32_t>(input.buffer()->address()),
        static_cast<uint32_t>(output.buffer()->address())};
    TT_FATAL(
        routes.size() <= 16,
        "persistent_moe_request: {} sends to {} chips, more than the kernel's 16",
        coord,
        routes.size());
    args.push_back(static_cast<uint32_t>(routes.size()));
    for (const auto& route : routes) {
        append_d2d_route_args(route, args);
    }
    // Two connections, each its own manager: the ack to the responder may leave in the request's direction,
    // which one manager cannot hold twice.
    tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
        src, {routes.front().next}, {0u}, desc, kernel_id, core, args, tt::tt_fabric::FabricApiType::Mesh);
    tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
        src, {endpoints->responder}, {1u}, desc, kernel_id, core, args, tt::tt_fabric::FabricApiType::Mesh);
    desc.kernels[kernel_id].runtime_args.emplace_back(
        core, KernelDescriptor::CoreRuntimeArgs(args.begin(), args.end()));
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
    auto& args = tt::tt_metal::GetRuntimeArgs(program, 0, endpoints->core);
    args[0] = attributes.layer;
    args[1] = input.buffer()->address();
    args[2] = output.buffer()->address();
}

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts

namespace ttnn::prim {

Tensor persistent_moe_request(
    const Tensor& input_tensor,
    uint32_t layer,
    const tt::tt_metal::distributed::MeshSocket& request_socket,
    const tt::tt_metal::distributed::MeshSocket& response_socket) {
    using OperationType =
        ttnn::operations::experimental::deepseek::moe::fused_experts::PersistentMoeRequestDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t(layer, request_socket, response_socket), input_tensor);
}

}  // namespace ttnn::prim
