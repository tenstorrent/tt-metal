// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bidirectional_fabric_socket.hpp"

#include <memory>

#include <nanobind/nanobind.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/shared_ptr.h>

#include <ttnn/distributed/isocket.hpp>
#include <ttnn/distributed/bidirectional_fabric_socket.hpp>

namespace ttnn::bidirectional_fabric_socket {

void py_module_types(nb::module_& mod) {
    nb::class_<ttnn::distributed::BidirectionalFabricSocket, ttnn::distributed::ISocket>(
        mod, "BidirectionalFabricSocket")
        .def(
            nb::init<const tt::tt_metal::distributed::MeshSocket&, const tt::tt_metal::distributed::MeshSocket&>(),
            nb::arg("send_socket"),
            nb::arg("recv_socket"),
            R"doc(
                Initialize a BidirectionalFabricSocket with send and receive MeshSockets.

                Bidirectional fabric-based implementation of distributed tensor communication
                that provides high-performance point-to-point tensor communication using
                Tenstorrent's fabric interconnect technology. This implementation leverages
                the underlying fabric hardware for direct chip-to-chip communication,
                offering lower latency and higher bandwidth compared to traditional network
                protocols.

                Args:
                    send_socket (MeshSocket): The mesh socket for sending
                    recv_socket (MeshSocket): The mesh socket for receiving
            )doc")
        .def_static(
            "create",
            &ttnn::distributed::BidirectionalFabricSocket::create,
            nb::arg("mesh_device"),
            nb::arg("rank"),
            nb::arg("socket_config"),
            R"doc(
                Create a BidirectionalFabricSocket for bidirectional communication with a specific rank.

                Args:
                    mesh_device (MeshDevice): The mesh device to use for communication
                    rank (Rank): The rank to communicate with
                    socket_config (SocketConfig): The socket configuration

                Returns:
                    BidirectionalFabricSocket: The created bidirectional fabric socket
            )doc");
}

void py_module(nb::module_& mod) {
    mod.def(
        "create",
        &ttnn::distributed::BidirectionalFabricSocket::create,
        nb::arg("mesh_device"),
        nb::arg("rank"),
        nb::arg("socket_config"),
        R"doc(
            Create a BidirectionalFabricSocket for bidirectional communication with a specific rank.

            Factory function to create a bidirectional fabric socket for simultaneous
            send and receive capabilities over Tenstorrent's fabric interconnect.

            Args:
                mesh_device (MeshDevice): The mesh device to use for communication
                rank (Rank): The rank to communicate with
                socket_config (SocketConfig): The socket configuration

            Returns:
                BidirectionalFabricSocket: The created bidirectional fabric socket
        )doc");
}

}  // namespace ttnn::bidirectional_fabric_socket
