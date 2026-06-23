// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "send_direct_async_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_direct_async/send_direct_async.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_send_direct_async(nb::module_& mod) {
    ttnn::bind_function<"send_direct_async", "ttnn.experimental.">(
        mod,
        R"doc(
        Performs a direct send of multi-device :attr:`input_tensor` to a :attr:`mesh_socket`.

        Unlike send_async, the sender writes each page straight into the receiver's output tensor
        instead of the socket FIFO. The socket is used only to advertise the sender handshake-buffer
        address (over which the receiver writes back its output tensor address) and to signal
        completion with a single page.

        Args:
            input_tensor (ttnn.Tensor): device tensor.
            mesh_socket (ttnn.MeshSocket): MeshSocket to send the tensor to.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Returns:
            std::vector<ttnn.Tensor>: an empty vector.

        )doc",
        &ttnn::experimental::send_direct_async,
        nb::arg("input_tensor"),
        nb::arg("mesh_socket"));
}

}  // namespace ttnn::operations::experimental::ccl
