// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async_h2d_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/recv_async_h2d/recv_async_h2d.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_recv_async_h2d(nb::module_& mod) {
    ttnn::bind_function<"recv_async_h2d", "ttnn.experimental.">(
        mod,
        R"doc(
        Performs a recv operation from an :attr:`h2d_socket` (Host-to-Device socket) into a
        pre-allocated device output tensor.

        Each page of the output tensor is received from the H2D socket FIFO and written
        directly into the corresponding tensor page on device. The H2D socket's mode
        (``HOST_PUSH`` vs ``DEVICE_PULL``) is honored at compile time:

        - ``HOST_PUSH``: the host writes data into device L1 directly; the receiver simply
          consumes the FIFO.
        - ``DEVICE_PULL``: the receiver issues PCIe NOC reads to pull data from pinned host
          memory into the FIFO before forwarding it to the output tensor.

        Args:
            output_tensor (ttnn.Tensor): Pre-allocated device tensor to receive the data into.
                The tensor's aligned page size must equal the H2D socket's page size and the
                socket FIFO must be at least one page in size.
            h2d_socket (ttnn.H2DSocket): H2DSocket to receive the data from. Must be the
                owner side (created via ``ttnn.H2DSocket(...)``, not via ``connect()``) so
                that a MeshDevice is available for program dispatch.

        Returns:
            List[ttnn.Tensor]: A list containing the (mutated) output tensor.

        )doc",
        &ttnn::experimental::recv_async_h2d,
        nb::arg("output_tensor"),
        nb::arg("h2d_socket"));
}

}  // namespace ttnn::operations::experimental::ccl
