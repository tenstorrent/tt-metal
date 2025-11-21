// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_async/send_async.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_async/device/send_async_op.hpp"
#include "ttnn/run_operation.hpp"
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
               const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
               const tt::tt_metal::distributed::SocketConfig& socket_config) -> std::vector<ttnn::Tensor> {
                return self(input_tensor, mesh_device, socket_config);
            },
            py::arg("input_tensor"),
            py::arg("mesh_device"),
            py::arg("socket_config")},
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const tt::tt_metal::distributed::MeshSocket& mesh_socket) -> std::vector<ttnn::Tensor> {
                return tt::tt_metal::operation::run(ttnn::SendAsync(mesh_socket), {input_tensor});
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
        Performs a send operation on multi-device :attr:`input_tensor` using :attr:`mesh_device` and :attr:`socket_config`.

        Args:
            input_tensor (ttnn.Tensor): device tensor.
            mesh_device (ttnn.MeshDevice): MeshDevice to send the tensor from.
            socket_config (ttnn.SocketConfig): SocketConfig containing socket connection and memory configuration.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Returns:
            std::vector<ttnn.Tensor>: an empty vector.

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
