// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_socket.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/pair.h>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

namespace ttnn::mesh_socket {

void py_module_types(nb::module_& mod) {
    nb::enum_<tt::tt_metal::distributed::SocketEndpoint>(mod, "SocketEndpoint")
        .value("SENDER", tt::tt_metal::distributed::SocketEndpoint::SENDER, "Sender endpoint")
        .value("RECEIVER", tt::tt_metal::distributed::SocketEndpoint::RECEIVER, "Receiver endpoint");

    nb::class_<tt::tt_metal::distributed::MeshCoreCoord>(mod, "MeshCoreCoord")
        .def(
            nb::init<tt::tt_metal::distributed::MeshCoordinate, tt::tt_metal::CoreCoord>(),
            nb::arg("device_coord"),
            nb::arg("core_coord"),
            R"doc(
                Initialize a MeshCoreCoord with device and core coordinates.

                Args:
                    device_coord (MeshCoordinate): The device coordinate of the core
                    core_coord (CoreCoord): The core coordinate of the core
                )doc")
        .def_rw("device_coord", &tt::tt_metal::distributed::MeshCoreCoord::device_coord, "Device coordinate")
        .def_rw("core_coord", &tt::tt_metal::distributed::MeshCoreCoord::core_coord, "Core coordinate")
        .def(
            "__eq__",
            [](const tt::tt_metal::distributed::MeshCoreCoord& a, const tt::tt_metal::distributed::MeshCoreCoord& b) {
                return a == b;
            })
        .def(
            "__ne__",
            [](const tt::tt_metal::distributed::MeshCoreCoord& a, const tt::tt_metal::distributed::MeshCoreCoord& b) {
                return a != b;
            })
        .def("__repr__", [](const tt::tt_metal::distributed::MeshCoreCoord& mcc) {
            std::stringstream ss;
            ss << "MeshCoreCoord(device=" << mcc.device_coord << ", core=(" << mcc.core_coord.x << ","
               << mcc.core_coord.y << "))";
            return ss.str();
        });
    nb::class_<tt::tt_metal::distributed::SocketConnection>(mod, "SocketConnection")
        .def(
            nb::init<tt::tt_metal::distributed::MeshCoreCoord, tt::tt_metal::distributed::MeshCoreCoord>(),
            nb::arg("sender_core"),
            nb::arg("receiver_core"),
            R"doc(
                Initialize a SocketConnection with sender and receiver core coordinates.

                Args:
                    sender_core (MeshCoreCoord): The mesh core coordinate of the sender
                    receiver_core (MeshCoreCoord): The mesh core coordinate of the receiver
            )doc")
        .def_rw("sender_core", &tt::tt_metal::distributed::SocketConnection::sender_core, "Sender core coordinate")
        .def_rw(
            "receiver_core", &tt::tt_metal::distributed::SocketConnection::receiver_core, "Receiver core coordinate");
    nb::class_<tt::tt_metal::distributed::SocketMemoryConfig>(mod, "SocketMemoryConfig")
        .def(
            nb::init<
                tt::tt_metal::BufferType,
                uint32_t,
                std::optional<tt::tt_metal::SubDeviceId>,
                std::optional<tt::tt_metal::SubDeviceId>>(),
            nb::arg("socket_storage_type"),
            nb::arg("fifo_size"),
            nb::arg("sender_sub_device") = nb::none(),
            nb::arg("receiver_sub_device") = nb::none(),
            R"doc(
                Initialize a SocketMemoryConfig with socket storage type, fifo size, sender sub device and receiver sub device.

                Args:
                    socket_storage_type (BufferType): The type of buffer to use for the socket
                    fifo_size (int): The size of the fifo
                    sender_sub_device (SubDeviceId, optional): The sub device of the sender
                    receiver_sub_device (SubDeviceId, optional): The sub device of the receiver
            )doc");
    nb::class_<tt::tt_metal::distributed::SocketConfig>(mod, "SocketConfig")
        .def(
            "__init__",
            [](tt::tt_metal::distributed::SocketConfig* config,
               std::vector<tt::tt_metal::distributed::SocketConnection> connections,
               tt::tt_metal::distributed::SocketMemoryConfig memory_config,
               std::optional<int> sender_rank,
               std::optional<int> receiver_rank) {
                new (config) tt::tt_metal::distributed::SocketConfig();
                config->socket_connection_config = std::move(connections);
                config->socket_mem_config = memory_config;
                if (sender_rank.has_value()) {
                    config->sender_rank = tt::tt_metal::distributed::multihost::Rank(sender_rank.value());
                }
                if (receiver_rank.has_value()) {
                    config->receiver_rank = tt::tt_metal::distributed::multihost::Rank(receiver_rank.value());
                }
            },
            nb::arg("connections"),
            nb::arg("memory_config"),
            nb::arg("sender_rank") = nb::none(),
            nb::arg("receiver_rank") = nb::none(),
            R"doc(
                Initialize a SocketConfig with connections and memory config.

                Args:
                    connections (List[SocketConnection]): The connections of the socket
                    memory_config (SocketMemoryConfig): The memory config of the socket
                    sender_rank (int, optional): The rank of the sender host in a multi-host context
                    receiver_rank (int, optional): The rank of the receiver host in a multi-host context
            )doc");
    nb::class_<tt::tt_metal::distributed::MeshSocket>(mod, "MeshSocket")
        .def(
            nb::init<
                const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>&,
                const tt::tt_metal::distributed::SocketConfig&>(),
            nb::arg("device"),
            nb::arg("config"),
            R"doc(
                Initialize a MeshSocket with a device and socket configuration.

                Args:
                    device (MeshDevice): The mesh device on which to create the socket
                    config (SocketConfig): The configuration for the socket

                Note:
                    Sockets should typically be created in pairs using create_socket_pair()
                    rather than using this constructor directly.
            )doc")
        .def(
            "get_config_buffer_address",
            [](const tt::tt_metal::distributed::MeshSocket& socket) {
                return static_cast<uint32_t>(socket.get_config_buffer()->address());
            },
            R"doc(
                Returns the L1 address of the socket configuration buffer on the device.
                This address is passed to device kernels to access socket metadata.
            )doc")
        .def(
            "get_active_cores",
            [](const tt::tt_metal::distributed::MeshSocket& socket) { return socket.get_active_cores(); },
            R"doc(
                Returns the active cores of the socket.
            )doc")
        .def(
            "get_mesh_device",
            [](const tt::tt_metal::distributed::MeshSocket& socket) { return socket.get_mesh_device(); },
            R"doc(
                Returns the mesh device of the socket.
            )doc")
        .def(
            "get_connection_config",
            [](const tt::tt_metal::distributed::MeshSocket& socket) {
                return socket.get_config().socket_connection_config;
            },
            R"doc(
            Returns the connection config of the socket.
            )doc")
        .def(
            "get_socket_endpoint_type",
            &tt::tt_metal::distributed::MeshSocket::get_socket_endpoint_type,
            R"doc(
                Returns the socket endpoint type (SENDER or RECEIVER).

                Returns:
                    SocketEndpoint: The endpoint type of this socket.
            )doc")
        .def(
            "get_fabric_node_id",
            &tt::tt_metal::distributed::MeshSocket::get_fabric_node_id,
            nb::arg("endpoint"),
            nb::arg("coord"),
            R"doc(
                Returns the fabric node ID for a given endpoint and device coordinate.

                Args:
                    endpoint (SocketEndpoint): The endpoint type (SENDER or RECEIVER).
                    coord (MeshCoordinate): The device coordinate to look up.

                Returns:
                    FabricNodeId: The fabric node ID for the given endpoint and coordinate.
            )doc");
}

void py_module(nb::module_& mod) {
    mod.def(
        "create_socket_pair",
        &tt::tt_metal::distributed::MeshSocket::create_socket_pair,
        nb::arg("sender_mesh_device"),
        nb::arg("receiver_mesh_device"),
        nb::arg("socket_config"),
        R"doc(
            Create a pair of sockets between two mesh devices.

            Args:
                sender_mesh_device (MeshDevice): The mesh device on which to create the sender socket.
                receiver_mesh_device (MeshDevice): The mesh device on which to create the receiver socket.
                socket_config (SocketConfig): The config of the socket pair.

            Returns:
                Tuple[MeshSocket, MeshSocket]: The pair of sockets.
            )doc");
}

}  // namespace ttnn::mesh_socket
