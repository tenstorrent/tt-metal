// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tt-metalium/mesh_socket.hpp>

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_recv_async(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& output_tensor,
               const tt::tt_metal::distributed::MeshSocket& mesh_socket) -> std::vector<ttnn::Tensor> {
                return self(output_tensor, mesh_socket);
            },
            py::arg("output_tensor"),
            py::arg("mesh_socket")});
}

}  // namespace

void py_bind_recv_async(pybind11::module& module) {
    bind_recv_async(
        module,
        ttnn::experimental::recv_async,
        R"doc(

        Performs a recv operation from a multi-device :attr:`mesh_socket`.

        Args:
            output_tensor (ttnn.Tensor): Tensor to receive the data into.
            mesh_socket (ttnn.MeshSocket): MeshSocket to receive the data from.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming%20Mesh%20of%20Devices/Programming%20Mesh%20of%20Devices%20with%20TT-NN.md

        Returns:
            std::vector<ttnn.Tensor>: A vector containing the output tensor.

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
