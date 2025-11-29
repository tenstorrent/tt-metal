// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
// #include <pybind11/trampoline_self_life_support.h>

#include "api/ttnn/distributed/bidirectional_fabric_socket.hpp"
#include "api/ttnn/distributed/create_socket.hpp"
#include "api/ttnn/distributed/fabric_socket.hpp"
#include "api/ttnn/distributed/isocket.hpp"
#include "api/ttnn/distributed/mpi_socket.hpp"
#include "ttnn-pybind/isocket.hpp"
#include "ttnn-pybind/export_enum.hpp"

namespace ttnn::distributed {

namespace py = pybind11;

class PyISocket : public ttnn::distributed::ISocket /*, public py::trampoline_self_life_support*/ {
public:
    using ttnn::distributed::ISocket::ISocket;

    void send(const ttnn::Tensor& tensor) override {
        PYBIND11_OVERRIDE_PURE(void, ttnn::distributed::ISocket, send, tensor);
    }

    void recv(ttnn::Tensor& tensor) override { PYBIND11_OVERRIDE_PURE(void, ttnn::distributed::ISocket, recv, tensor); }

    tt::tt_metal::distributed::multihost::Rank get_rank() const override {
        PYBIND11_OVERRIDE_PURE(
            tt::tt_metal::distributed::multihost::Rank, ttnn::distributed::ISocket, get_rank, /*no args*/);
    }

    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> get_distributed_context() const override {
        PYBIND11_OVERRIDE_PURE(
            std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>,
            ttnn::distributed::ISocket,
            get_distributed_context,
            /*no args*/);
    }
};  // class PyISocket

namespace {

auto constexpr send_doc_string = R"doc(
    Send a Tensor.

    Keyword Args:
        tensor (Tensor): The tensor to send.
)doc";

auto constexpr recv_doc_string = R"doc(
    Receive a Tensor.

    Keyword Args:
        tensor (Tensor): The tensor to hold the received data.
)doc";

auto constexpr get_rank_doc_string = "Get Socket Rank.";
auto constexpr get_distributed_context_doc_string = "Get Distributed Context.";
auto constexpr tensor_arg_string = "tensor";

}  // namespace

void py_isocket_module_types(py::module& module) {
    export_enum<ttnn::distributed::SocketType>(module, "SocketType");
    export_enum<ttnn::distributed::EndpointSocketType>(module, "EndpointSocketType");

    py::class_<ttnn::distributed::ISocket, PyISocket /*, py::smart_holder*/>(module, "ISocket")
        .def("send", &ISocket::send, py::arg(tensor_arg_string), send_doc_string)
        .def("recv", &ISocket::recv, py::arg(tensor_arg_string), recv_doc_string)
        .def("get_rank", &ISocket::get_rank, get_rank_doc_string)
        .def("get_distributed_context", &ISocket::get_distributed_context, get_distributed_context_doc_string);

    py::class_<ttnn::distributed::BidirectionalFabricSocket, ttnn::distributed::ISocket /*, py::smart_holder*/>(
        module, "BidirectionalFabricSocket")
        .def(py::init<
             const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>&,
             const tt::tt_metal::distributed::SocketConfig&,
             const tt::tt_metal::distributed::SocketConfig&>())
        .def_static("create", &BidirectionalFabricSocket::create)
        .def("send", &ttnn::distributed::BidirectionalFabricSocket::send, py::arg(tensor_arg_string), send_doc_string)
        .def("recv", &ttnn::distributed::BidirectionalFabricSocket::recv, py::arg(tensor_arg_string), recv_doc_string)
        .def("get_rank", &ttnn::distributed::BidirectionalFabricSocket::get_rank, get_rank_doc_string)
        .def(
            "get_distributed_context",
            &ttnn::distributed::BidirectionalFabricSocket::get_distributed_context,
            get_distributed_context_doc_string);

    py::class_<ttnn::distributed::FabricSocket, ttnn::distributed::ISocket /*, py::smart_holder*/>(
        module, "FabricSocket")
        .def(py::init<
             const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>&,
             const tt::tt_metal::distributed::SocketConfig&>())
        .def_static("create", &FabricSocket::create)
        .def("send", &ttnn::distributed::FabricSocket::send, py::arg(tensor_arg_string), send_doc_string)
        .def("recv", &ttnn::distributed::FabricSocket::recv, py::arg(tensor_arg_string), recv_doc_string)
        .def("get_rank", &ttnn::distributed::FabricSocket::get_rank, get_rank_doc_string)
        .def(
            "get_distributed_context",
            &ttnn::distributed::FabricSocket::get_distributed_context,
            get_distributed_context_doc_string);

    py::class_<ttnn::distributed::MPISocket, ttnn::distributed::ISocket /*, py::smart_holder*/>(module, "MPISocket")
        .def(py::init<
             const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>&,
             const tt::tt_metal::distributed::SocketConfig&>())
        .def_static("create", &MPISocket::create)
        .def("send", &ttnn::distributed::MPISocket::send, py::arg(tensor_arg_string), send_doc_string)
        .def("recv", &ttnn::distributed::MPISocket::recv, py::arg(tensor_arg_string), recv_doc_string)
        .def("get_rank", &ttnn::distributed::MPISocket::get_rank, get_rank_doc_string)
        .def(
            "get_distributed_context",
            &ttnn::distributed::MPISocket::get_distributed_context,
            get_distributed_context_doc_string);

    module.def(
        "create_socket",
        [](ttnn::distributed::SocketType socket_type,
           ttnn::distributed::EndpointSocketType endpoint_socket_type,
           std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
           int other_rank,
           const tt::tt_metal::distributed::SocketConfig& socket_config) {
            return ttnn::distributed::create_socket(
                socket_type,
                endpoint_socket_type,
                mesh_device,
                tt::tt_metal::distributed::multihost::Rank(other_rank),
                socket_config);
        },
        py::arg("socket_type"),
        py::arg("endpoint_socket_type"),
        py::arg("mesh_device"),
        py::arg("other_rank"),
        py::arg("socket_config"));
}

}  // namespace ttnn::distributed
