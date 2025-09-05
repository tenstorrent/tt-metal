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
        PYBIND11_OVERRIDE_PURE(tt::tt_metal::distributed::multihost::Rank, ttnn::distributed::ISocket, get_rank, );
    }

    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> get_distributed_context() const override {
        PYBIND11_OVERRIDE_PURE(
            std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>,
            ttnn::distributed::ISocket,
            get_distributed_context, );
    }
};  // class PyISocket

void py_isocket_module_types(py::module& module) {
    py::class_<ttnn::distributed::ISocket, PyISocket /*, py::smart_holder*/>(module, "ISocket")
        .def(py::init<>())
        .def("send", &ISocket::send)
        .def("recv", &ISocket::recv)
        .def("get_rank", &ISocket::get_rank)
        .def("get_distributed_context", &ISocket::get_distributed_context);

    py::class_<ttnn::distributed::BidirectionalFabricSocket, ttnn::distributed::ISocket /*, py::smart_holder*/>(
        module, "BidirectionalFabricSocket")
        .def(py::init<tt::tt_metal::distributed::MeshSocket&, tt::tt_metal::distributed::MeshSocket&>())
        .def("send", &BidirectionalFabricSocket::send)
        .def("recv", &BidirectionalFabricSocket::recv)
        .def("get_rank", &BidirectionalFabricSocket::get_rank)
        .def("get_distributed_context", &BidirectionalFabricSocket::get_distributed_context)
        .def_static("create", &BidirectionalFabricSocket::create);

    py::class_<ttnn::distributed::FabricSocket, ttnn::distributed::ISocket /*, py::smart_holder*/>(
        module, "FabricSocket")
        .def(py::init<tt::tt_metal::distributed::MeshSocket&>())
        .def("send", &FabricSocket::send)
        .def("recv", &FabricSocket::recv)
        .def("get_rank", &FabricSocket::get_rank)
        .def("get_distributed_context", &FabricSocket::get_distributed_context)
        .def_static("create", &FabricSocket::create);

    py::class_<ttnn::distributed::MPISocket, ttnn::distributed::ISocket /*, py::smart_holder*/>(module, "MPISocket")
        .def(py::init<tt::tt_metal::distributed::MeshSocket&>())
        .def("send", &MPISocket::send)
        .def("recv", &MPISocket::recv)
        .def("get_rank", &MPISocket::get_rank)
        .def("get_distributed_context", &MPISocket::get_distributed_context)
        .def_static("create", &MPISocket::create);
}

}  // namespace ttnn::distributed
