// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_direct_async_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/recv_direct_async/recv_direct_async.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_recv_direct_async(nb::module_& mod) {
    ttnn::bind_function<"recv_direct_async", "ttnn.experimental.">(
        mod,
        R"doc(
        Receives a tensor sent by send_direct_async.

        The sender writes :attr:`output_tensor` directly, so this op only advertises its address over
        the socket and waits for the completion signal.

        Args:
            output_tensor (ttnn.Tensor): Tensor to receive the data into.
            mesh_socket (ttnn.MeshSocket): MeshSocket to receive the data from.

        Keyword Args:
            wait_only (bool): skip the address write-back and only wait for the completion page.
                Pair with send_direct_async(..., static_dst_address=output_tensor.buffer_address()).
                Defaults to False.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Returns:
            std::vector<ttnn.Tensor>: A vector containing the output tensor.

        )doc",
        &ttnn::experimental::recv_direct_async,
        nb::arg("output_tensor"),
        nb::arg("mesh_socket"),
        nb::kw_only(),
        nb::arg("wait_only") = false);
}

}  // namespace ttnn::operations::experimental::ccl
