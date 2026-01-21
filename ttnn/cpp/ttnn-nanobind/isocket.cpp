// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "isocket.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/trampoline.h>

#include <ttnn/distributed/isocket.hpp>
#include <ttnn/distributed/create_socket.hpp>

namespace ttnn::isocket {

/**
 * Trampoline class that enables Python subclassing of ISocket.
 *
 * This allows users to implement custom socket behavior in pure Python:
 *
 *     class MyPythonSocket(ISocket):
 *         def __init__(self):
 *             super().__init__()
 *             self.buffer = None
 *
 *         def send(self, tensor):
 *             self.buffer = tensor.cpu()
 *
 *         def recv(self, tensor):
 *             # Copy buffer to tensor
 *             pass
 *
 *         def get_rank(self):
 *             return 0
 *
 *         def get_distributed_context(self):
 *             return None
 */
class PyISocket : public ttnn::distributed::ISocket {
public:
    NB_TRAMPOLINE(ttnn::distributed::ISocket, 4);

    void send(const ttnn::Tensor& tensor) override { NB_OVERRIDE_PURE(send, tensor); }

    void recv(ttnn::Tensor& tensor) override { NB_OVERRIDE_PURE(recv, tensor); }

    tt::tt_metal::distributed::multihost::Rank get_rank() const override { NB_OVERRIDE_PURE(get_rank); }

    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> get_distributed_context() const override {
        NB_OVERRIDE_PURE(get_distributed_context);
    }
};

void py_module_types(nb::module_& mod) {
    nb::enum_<ttnn::distributed::SocketType>(mod, "SocketType")
        .value("MPI", ttnn::distributed::SocketType::MPI)
        .value("FABRIC", ttnn::distributed::SocketType::FABRIC);

    nb::enum_<ttnn::distributed::EndpointSocketType>(mod, "EndpointSocketType")
        .value("SENDER", ttnn::distributed::EndpointSocketType::SENDER)
        .value("RECEIVER", ttnn::distributed::EndpointSocketType::RECEIVER)
        .value("BIDIRECTIONAL", ttnn::distributed::EndpointSocketType::BIDIRECTIONAL);

    nb::class_<ttnn::distributed::ISocket, PyISocket>(mod, "ISocket")
        .def(nb::init<>())
        .def(
            "send",
            &ttnn::distributed::ISocket::send,
            nb::arg("tensor"),
            R"doc(
                Send a tensor over the socket.

                Args:
                    tensor (Tensor): The tensor to send
            )doc")
        .def(
            "recv",
            &ttnn::distributed::ISocket::recv,
            nb::arg("tensor"),
            R"doc(
                Receive a tensor over the socket.

                Args:
                    tensor (Tensor): The tensor to receive into
            )doc")
        .def(
            "get_rank",
            &ttnn::distributed::ISocket::get_rank,
            R"doc(
                Get the rank associated with this socket.

                Returns:
                    Rank: The rank of this socket endpoint
            )doc")
        .def(
            "get_distributed_context",
            &ttnn::distributed::ISocket::get_distributed_context,
            R"doc(
                Get the distributed context associated with this socket.

                Returns:
                    DistributedContext: The distributed context
            )doc");
}

void py_module(nb::module_& mod) {
    mod.def(
        "create_socket",
        &ttnn::distributed::create_socket,
        nb::arg("socket_type"),
        nb::arg("endpoint_socket_type"),
        nb::arg("mesh_device"),
        nb::arg("other_rank"),
        nb::arg("socket_config"),
        R"doc(
            Create a socket for distributed tensor communication.

            Factory function to create the appropriate socket type based on the specified
            socket type and endpoint type.

            Args:
                socket_type (SocketType): The type of socket to create (MPI or FABRIC)
                endpoint_socket_type (EndpointSocketType): The endpoint type (SENDER, RECEIVER, or BIDIRECTIONAL)
                mesh_device (MeshDevice): The mesh device to use for communication
                other_rank (Rank): The rank of the other endpoint to communicate with
                socket_config (SocketConfig): The socket configuration

            Returns:
                ISocket: The created socket instance
        )doc");
}

}  // namespace ttnn::isocket
