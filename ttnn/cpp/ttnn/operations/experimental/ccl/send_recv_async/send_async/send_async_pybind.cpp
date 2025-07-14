// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_async/send_async.hpp"
#include <tt-metalium/mesh_socket.hpp>

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_send_async(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const tt::tt_metal::distributed::MeshSocket& mesh_socket) -> std::vector<ttnn::Tensor> {
                return self(input_tensor, mesh_socket);
            },
            py::arg("input_tensor"),
            py::arg("mesh_socket")});
}

}  // namespace

void py_bind_send_async(pybind11::module& module) {
    bind_send_async(
        module,
        ttnn::experimental::send_async,
        R"doc(

        Performs a send operation on multi-device :attr:`input_tensor` to a :attr:`mesh_socket`.

        Args:
            input_tensor (ttnn.Tensor): device tensor.
            mesh_socket (ttnn.MeshSocket): MeshSocket to send the tensor to.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Returns:
            std::vector<ttnn.Tensor>: an empty vector.

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
