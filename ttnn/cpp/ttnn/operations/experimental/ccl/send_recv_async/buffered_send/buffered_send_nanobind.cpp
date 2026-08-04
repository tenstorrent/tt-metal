// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
        Sends :attr:`input_tensor` over :attr:`mesh_socket`, paired with buffered_recv.

        Like send_direct_async, pages are written straight into the receiver's memory rather than
        through the socket FIFO; the destination is whichever of buffered_recv's output tensors is
        free, so the sender blocks while the ring is full.

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
