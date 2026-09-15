// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_direct_async_op_program_factory.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {

// The socket connections whose receiver core sits on `target_device`, in socket-connection order.
// create_descriptor and override_runtime_arguments both walk this so the per-core runtime-arg
// ordering they assume stays identical.
struct ReceiverConnections {
    std::vector<CoreCoord> core_coords;
    std::vector<tt::tt_fabric::FabricNodeId> sender_fabric_node_ids;
    std::vector<tt::tt_fabric::FabricNodeId> receiver_fabric_node_ids;
};

ReceiverConnections collect_receiver_connections(
    const tt::tt_metal::distributed::MeshSocket& mesh_socket, const Tensor& output_tensor, IDevice* target_device) {
    const auto* socket_mesh_device = mesh_socket.get_config_buffer()->device();
    ReceiverConnections connections;
    for (const auto& connection : mesh_socket.get_config().socket_connection_config) {
        if (socket_mesh_device->get_device(connection.receiver_core.device_coord)->id() == target_device->id()) {
            connections.core_coords.push_back(connection.receiver_core.core_coord);
            connections.receiver_fabric_node_ids.push_back(
                output_tensor.device()->get_fabric_node_id(connection.receiver_core.device_coord));
            connections.sender_fabric_node_ids.push_back(mesh_socket.get_fabric_node_id(
                tt::tt_metal::distributed::SocketEndpoint::SENDER, connection.sender_core.device_coord));
        }
    }
    return connections;
}

// Descriptor kernel index, fixed by the push order at the end of create_descriptor.
constexpr uint32_t handshake_kernel_index = 0;

}  // namespace

ProgramDescriptor RecvDirectAsyncProgramFactory::create_descriptor(
    const RecvDirectAsyncParams& operation_attributes,
    const Tensor& tensor_args,
    std::vector<Tensor>& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    const auto& mesh_socket = operation_attributes.mesh_socket;
    const auto& output_tensor = tensor_args;
    IDevice* target_device =
        ttnn::send_recv_utils::resolve_target_device(output_tensor, mesh_dispatch_coordinate, "recv_direct_async");

    auto connections = collect_receiver_connections(mesh_socket, output_tensor, target_device);
    const auto& receiver_core_coords = connections.core_coords;
    const auto& sender_fabric_node_ids = connections.sender_fabric_node_ids;
    const auto& receiver_fabric_node_ids = connections.receiver_fabric_node_ids;

    uint32_t num_cores = receiver_core_coords.size();
    // This device holds no receiver core of the socket, so it has no work. An empty descriptor tells
    // the framework to skip this coordinate.
    if (num_cores == 0) {
        return ProgramDescriptor{};
    }

    // cores must not exceed available fabric links
    {
        auto available_link_indices =
            tt::tt_fabric::get_forwarding_link_indices(receiver_fabric_node_ids[0], sender_fabric_node_ids[0]);
        uint32_t num_available_links = available_link_indices.size();

        TT_FATAL(
            num_cores <= num_available_links,
            "Cannot create {} receiver-sender pairs with only {} available fabric links between devices. "
            "Reduce the number of cores per device.",
            num_cores,
            num_available_links);
    }

    auto max_alignment = std::max(
        target_device->allocator()->get_alignment(mesh_socket.get_config().socket_mem_config.socket_storage_type),
        output_tensor.buffer()->alignment());

    uint32_t handshake_page_size = ttnn::send_recv_utils::handshake_page_size(max_alignment);

    auto receiver_core_range_set = CoreRangeSet(std::set<CoreRange>());
    for (const auto& core : receiver_core_coords) {
        receiver_core_range_set = receiver_core_range_set.merge(CoreRangeSet({CoreRange(core, core)}));
    }

    ProgramDescriptor desc;

    constexpr uint8_t packet_header_cb_index = tt::CBIndex::c_0;
    uint32_t packet_header_cb_num_pages = 2;
    uint32_t packet_header_cb_page_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    desc.cbs.push_back(CBDescriptor{
        .total_size = packet_header_cb_num_pages * packet_header_cb_page_size,
        .core_ranges = receiver_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = packet_header_cb_index,
            .data_format = tt::DataFormat::UInt32,
            .page_size = packet_header_cb_page_size,
        }}},
    });

    KernelDescriptor handshake;
    handshake.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_direct_async/device/kernels/"
        "receiver_direct.cpp";
    handshake.source_type = KernelDescriptor::SourceType::FILE_PATH;
    handshake.core_ranges = receiver_core_range_set;
    handshake.compile_time_args = {
        packet_header_cb_index,  // fabric_packet_header_cb_id
        handshake_page_size,     // handshake_page_size (socket page size)
    };
    handshake.config = WriterConfigDescriptor{};

    for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        const auto& receiver_core_coord = receiver_core_coords[core_idx];
        const auto& sender_fabric_node_id = sender_fabric_node_ids[core_idx];
        const auto& receiver_fabric_node_id = receiver_fabric_node_ids[core_idx];

        // Both addresses are re-applied by override_runtime_arguments instead of being declared as
        // Buffer* bindings: the socket config buffer is not tensor-backed, so the binding fast path
        // would patch the output address and leave the socket address frozen at first miss.
        std::vector<uint32_t> handshake_rt_args = {
            mesh_socket.get_config_buffer()->address(),  // socket_config_addr
            output_tensor.buffer()->address(),           // output_base_addr
        };

        auto link_indices = tt::tt_fabric::get_forwarding_link_indices(receiver_fabric_node_id, sender_fabric_node_id);
        TT_FATAL(!link_indices.empty(), "No link indices found for receiver core");

        uint32_t selected_link_index = link_indices[core_idx % link_indices.size()];
        tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
            receiver_fabric_node_id,
            sender_fabric_node_id,
            selected_link_index,
            desc,
            receiver_core_coord,
            handshake_rt_args);

        handshake.runtime_args.emplace_back(receiver_core_coord, std::move(handshake_rt_args));
    }

    desc.kernels.push_back(std::move(handshake));

    return desc;
}

void RecvDirectAsyncProgramFactory::override_runtime_arguments(
    Program& program,
    const RecvDirectAsyncParams& operation_attributes,
    const Tensor& tensor_args,
    std::vector<Tensor>& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    const auto& mesh_socket = operation_attributes.mesh_socket;
    const auto& output_tensor = tensor_args;
    IDevice* target_device =
        ttnn::send_recv_utils::resolve_target_device(output_tensor, mesh_dispatch_coordinate, "recv_direct_async");

    // The rest of the runtime args (the fabric connection trailer) derives from the socket topology,
    // which is in the program hash — so on a cache hit only these two base addresses can have moved.
    const uint32_t socket_config_addr = mesh_socket.get_config_buffer()->address();
    const uint32_t output_base_addr = output_tensor.buffer()->address();

    for (const auto& receiver_core_coord :
         collect_receiver_connections(mesh_socket, output_tensor, target_device).core_coords) {
        auto& handshake_runtime_args = GetRuntimeArgs(program, handshake_kernel_index, receiver_core_coord);
        handshake_runtime_args[0] = socket_config_addr;
        handshake_runtime_args[1] = output_base_addr;
    }
}

}  // namespace ttnn::experimental::prim
