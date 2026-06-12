// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "buffered_send_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/buffered_send/buffered_send.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_buffered_send(nb::module_& mod) {
    ttnn::bind_function<"buffered_send", "ttnn.experimental.">(
        mod,
        R"doc(
        Performs a buffered send of multi-device :attr:`input_tensor` to a :attr:`mesh_socket`.

        Behaves the same as send_direct_async: the sender writes each page straight into the
        receiver's output tensor instead of the socket FIFO. The socket is used only to advertise
        the sender handshake-buffer address (over which the receiver writes back its output tensor
        address) and to signal completion with a single page.

        Args:
            input_tensor (ttnn.Tensor): device tensor.
            mesh_socket (ttnn.MeshSocket): MeshSocket to send the tensor to.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Returns:
            std::vector<ttnn.Tensor>: an empty vector.

        )doc",
        &ttnn::experimental::buffered_send,
        nb::arg("input_tensor"),
        nb::arg("mesh_socket"));
}

}  // namespace ttnn::operations::experimental::ccl
