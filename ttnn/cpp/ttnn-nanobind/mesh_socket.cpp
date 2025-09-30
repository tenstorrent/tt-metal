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

#include <tt-metalium/mesh_socket.hpp>

namespace ttnn::mesh_socket {

void py_module_types(nb::module_& mod) {
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
            )doc");
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
            )doc");
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
            nb::init<
                std::vector<tt::tt_metal::distributed::SocketConnection>,
                tt::tt_metal::distributed::SocketMemoryConfig>(),
            nb::arg("connections"),
            nb::arg("memory_config"),
            R"doc(
                Initialize a SocketConfig with connections and memory config.

                Args:
                    connections (List[SocketConnection]): The connections of the socket
                    memory_config (SocketMemoryConfig): The memory config of the socket
            )doc");
    nb::class_<tt::tt_metal::distributed::MeshSocket>(mod, "MeshSocket");
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
