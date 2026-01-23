// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_socket.hpp"

#include <memory>

#include <nanobind/nanobind.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/shared_ptr.h>

#include <ttnn/distributed/isocket.hpp>
#include <ttnn/distributed/fabric_socket.hpp>

namespace ttnn::fabric_socket {

void py_module_types(nb::module_& mod) {
    nb::class_<ttnn::distributed::FabricSocket, ttnn::distributed::ISocket>(mod, "FabricSocket")
        .def(
            nb::init<const tt::tt_metal::distributed::MeshSocket&>(),
            nb::arg("mesh_socket"),
            R"doc(
                Initialize a FabricSocket with a MeshSocket.

                Fabric-based implementation of distributed tensor communication that provides
                high-performance point-to-point tensor communication using Tenstorrent's
                fabric interconnect technology. This implementation leverages the underlying
                fabric hardware for direct chip-to-chip communication, offering lower latency
                and higher bandwidth compared to traditional network protocols.

                Args:
                    mesh_socket (MeshSocket): The underlying mesh socket for communication
            )doc")
        .def_static(
            "create",
            &ttnn::distributed::FabricSocket::create,
            nb::arg("mesh_device"),
            nb::arg("sender_rank"),
            nb::arg("receiver_rank"),
            nb::arg("socket_config"),
            R"doc(
                Create a FabricSocket for communication between sender and receiver ranks.

                Args:
                    mesh_device (MeshDevice): The mesh device to use for communication
                    sender_rank (Rank): The sender rank
                    receiver_rank (Rank): The receiver rank
                    socket_config (SocketConfig): The socket configuration

                Returns:
                    FabricSocket: The created fabric socket
            )doc");
}

void py_module(nb::module_& mod) {
    mod.def(
        "create",
        &ttnn::distributed::FabricSocket::create,
        nb::arg("mesh_device"),
        nb::arg("sender_rank"),
        nb::arg("receiver_rank"),
        nb::arg("socket_config"),
        R"doc(
            Create a FabricSocket for communication between sender and receiver ranks.

            Factory function to create a fabric socket for high-performance point-to-point
            tensor communication using Tenstorrent's fabric interconnect.

            Args:
                mesh_device (MeshDevice): The mesh device to use for communication
                sender_rank (Rank): The sender rank
                receiver_rank (Rank): The receiver rank
                socket_config (SocketConfig): The socket configuration

            Returns:
                FabricSocket: The created fabric socket
        )doc");
}

}  // namespace ttnn::fabric_socket
