// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async_nanobind.hpp"

#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/recv_async/recv_async.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_recv_async(nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& output_tensor,
               const tt::tt_metal::distributed::MeshSocket& mesh_socket) -> std::vector<ttnn::Tensor> {
                return self(output_tensor, mesh_socket);
            },
            nb::arg("output_tensor"),
            nb::arg("mesh_socket")});
}

}  // namespace

void bind_recv_async(nb::module_& mod) {
    bind_recv_async(
        mod,
        ttnn::experimental::recv_async,
        R"doc(
        Performs a recv operation from a :attr:`mesh_socket`.

        Args:
            output_tensor (ttnn.Tensor): Tensor to receive the data into.
            mesh_socket (ttnn.MeshSocket): MeshSocket to receive the data from.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Returns:
            std::vector<ttnn.Tensor>: A vector containing the output tensor.

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
