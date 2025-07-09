// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/send_async/send_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/mesh_socket.hpp>

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_send_async(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    // namespace py = pybind11;

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

        Performs an send operation on multi-device :attr:`input_tensor` to a multi-device :attr:`mesh_socket`.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor.
            mesh_socket (ttnn.MeshSocket): MeshSocket to send the tensor to.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming%20Mesh%20of%20Devices/Programming%20Mesh%20of%20Devices%20with%20TT-NN.md

        Returns:
            std::vector<ttnn.Tensor>: a vector of tensors from all the devices.

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
