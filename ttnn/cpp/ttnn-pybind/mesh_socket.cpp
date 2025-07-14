// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_socket.hpp"

#include <cstdint>
#include <memory>
#include <utility>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tt-metalium/mesh_socket.hpp>

namespace ttnn::mesh_socket {

void py_module_types(py::module& module) {
    py::class_<tt::tt_metal::distributed::MeshCoreCoord>(module, "MeshCoreCoord")
        .def(
            py::init<tt::tt_metal::distributed::MeshCoordinate, tt::tt_metal::CoreCoord>(),
            py::arg("device_coord"),
            py::arg("core_coord"),
            R"doc(
                Initialize a MeshCoreCoord with device and core coordinates.

                Args:
                    device_coord (MeshCoordinate): The device coordinate of the core
                    core_coord (CoreCoord): The core coordinate of the core
            )doc");
    py::class_<tt::tt_metal::distributed::SocketConnection>(module, "SocketConnection")
        .def(
            py::init<tt::tt_metal::distributed::MeshCoreCoord, tt::tt_metal::distributed::MeshCoreCoord>(),
            py::arg("sender_core"),
            py::arg("receiver_core"),
            R"doc(
                Initialize a SocketConnection with sender and receiver core coordinates.

                Args:
                    sender_core (MeshCoreCoord): The mesh core coordinate of the sender
                    receiver_core (MeshCoreCoord): The mesh core coordinate of the receiver
            )doc");
    py::class_<tt::tt_metal::distributed::SocketMemoryConfig>(module, "SocketMemoryConfig")
        .def(
            py::init<
                tt::tt_metal::BufferType,
                uint32_t,
                std::optional<tt::tt_metal::SubDeviceId>,
                std::optional<tt::tt_metal::SubDeviceId>>(),
            py::arg("socket_storage_type"),
            py::arg("fifo_size"),
            py::arg("sender_sub_device") = std::nullopt,
            py::arg("receiver_sub_device") = std::nullopt,
            R"doc(
                Initialize a SocketMemoryConfig with socket storage type, fifo size, sender sub device and receiver sub device.

                Args:
                    socket_storage_type (BufferType): The type of buffer to use for the socket
                    fifo_size (int): The size of the fifo
                    sender_sub_device (SubDeviceId, optional): The sub device of the sender
                    receiver_sub_device (SubDeviceId, optional): The sub device of the receiver
            )doc");
    py::class_<tt::tt_metal::distributed::SocketConfig>(module, "SocketConfig")
        .def(
            py::init<
                std::vector<tt::tt_metal::distributed::SocketConnection>,
                tt::tt_metal::distributed::SocketMemoryConfig>(),
            py::arg("connections"),
            py::arg("memory_config"),
            R"doc(
                Initialize a SocketConfig with connections and memory config.

                Args:
                    connections (List[SocketConnection]): The connections of the socket
                    memory_config (SocketMemoryConfig): The memory config of the socket
            )doc");
    py::class_<tt::tt_metal::distributed::MeshSocket, std::shared_ptr<tt::tt_metal::distributed::MeshSocket>>(
        module, "MeshSocket");
}

void py_module(py::module& module) {
    module.def(
        "create_socket_pair",
        &tt::tt_metal::distributed::MeshSocket::create_socket_pair,
        py::arg("sender_mesh_device"),
        py::arg("receiver_mesh_device"),
        py::arg("socket_config"),
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
