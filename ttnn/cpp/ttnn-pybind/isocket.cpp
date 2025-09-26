// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
// #include <pybind11/trampoline_self_life_support.h>

#include "api/ttnn/distributed/bidirectional_fabric_socket.hpp"
#include "api/ttnn/distributed/fabric_socket.hpp"
#include "api/ttnn/distributed/isocket.hpp"
#include "api/ttnn/distributed/mpi_socket.hpp"
#include "ttnn-pybind/isocket.hpp"

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
    py::class_<ttnn::distributed::ISocket, PyISocket /*, py::smart_holder*/>(module, "ISocket")
        .def(py::init<>())
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
        .def("send", &BidirectionalFabricSocket::send, py::arg(tensor_arg_string), send_doc_string)
        .def("recv", &BidirectionalFabricSocket::recv, py::arg(tensor_arg_string), recv_doc_string)
        .def("get_rank", &BidirectionalFabricSocket::get_rank, get_rank_doc_string)
        .def(
            "get_distributed_context",
            &BidirectionalFabricSocket::get_distributed_context,
            get_distributed_context_doc_string)
        .def_static("create", &BidirectionalFabricSocket::create);

    py::class_<ttnn::distributed::FabricSocket, ttnn::distributed::ISocket /*, py::smart_holder*/>(
        module, "FabricSocket")
        .def(py::init<
             const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>&,
             const tt::tt_metal::distributed::SocketConfig&>())
        .def("send", &FabricSocket::send, py::arg(tensor_arg_string), send_doc_string)
        .def("recv", &FabricSocket::recv, py::arg(tensor_arg_string), recv_doc_string)
        .def("get_rank", &FabricSocket::get_rank, get_rank_doc_string)
        .def("get_distributed_context", &FabricSocket::get_distributed_context, get_distributed_context_doc_string)
        .def_static("create", &FabricSocket::create);

    py::class_<ttnn::distributed::MPISocket, ttnn::distributed::ISocket /*, py::smart_holder*/>(module, "MPISocket")
        .def(py::init<
             const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>&,
             const tt::tt_metal::distributed::SocketConfig&>())
        .def("send", &MPISocket::send, py::arg(tensor_arg_string), send_doc_string)
        .def("recv", &MPISocket::recv, py::arg(tensor_arg_string), recv_doc_string)
        .def("get_rank", &MPISocket::get_rank, get_rank_doc_string)
        .def("get_distributed_context", &MPISocket::get_distributed_context, get_distributed_context_doc_string)
        .def_static("create", &MPISocket::create);
}

}  // namespace ttnn::distributed
