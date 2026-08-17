// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async_d2h_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_async_d2h/send_async_d2h.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_send_async_d2h(nb::module_& mod) {
    ttnn::bind_function<"send_async_d2h", "ttnn.experimental.">(
        mod,
        R"doc(
        Performs a send operation from an :attr:`input_tensor` on device into a
        :attr:`d2h_socket` (Device-to-Host socket).

        Each page of the input tensor is staged in device L1 and then written out to the
        D2H socket FIFO in pinned host memory via PCIe NOC writes. The host can then
        consume the data by calling ``D2HSocket.read()`` (or ``read_tensor()``).

        The input tensor's aligned page size must equal the D2H socket's page size, and
        the socket FIFO must be at least one page in size. The D2H socket must be the
        owner side (created via ``ttnn.D2HSocket(...)``, not via ``connect()``) so that
        a MeshDevice is available for program dispatch.

        Args:
            input_tensor (ttnn.Tensor): Device tensor whose pages will be streamed to the
                host. The tensor's aligned page size must equal the D2H socket's page
                size.
            d2h_socket (ttnn.D2HSocket): D2HSocket to send the data into.

        Returns:
            List[ttnn.Tensor]: An empty list. The op does not produce any output tensors;
            data is delivered to the host via the D2H socket.

        )doc",
        &ttnn::experimental::send_async_d2h,
        nb::arg("input_tensor"),
        nb::arg("d2h_socket"));
}

}  // namespace ttnn::operations::experimental::ccl
