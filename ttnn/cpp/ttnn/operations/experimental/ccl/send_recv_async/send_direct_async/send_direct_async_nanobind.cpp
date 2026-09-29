// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
        Sends :attr:`input_tensor` over :attr:`mesh_socket`, paired with recv_direct_async.

        Unlike send_async, pages are written straight into the receiver's output tensor rather than
        through the socket FIFO, which carries only the address exchange and the completion signal.

        Args:
            input_tensor (ttnn.Tensor): device tensor.
            mesh_socket (ttnn.MeshSocket): MeshSocket to send the tensor to.

        Keyword Args:
            static_dst_address (int, optional): receiver output-buffer base address. When given, the
                address handshake is skipped (fire-and-forget): the sender waits only for socket FIFO
                credit, streams the pages to this address, then pushes the completion page. The
                receiver must call recv_direct_async(..., wait_only=True). Defaults to None.
            num_pages (int, optional): send only the first num_pages pages of the input buffer (e.g. the
                leading block prefix of a paged KV cache), written to the same page indices at the
                receiver. Defaults to None (all pages).

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Returns:
            std::vector<ttnn.Tensor>: an empty vector.

        )doc",
        &ttnn::experimental::send_direct_async,
        nb::arg("input_tensor"),
        nb::arg("mesh_socket"),
        nb::kw_only(),
        nb::arg("static_dst_address") = nb::none(),
        nb::arg("num_pages") = nb::none());
}

}  // namespace ttnn::operations::experimental::ccl
