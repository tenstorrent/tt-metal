// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/recv_async/recv_async.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_recv_async(nb::module_& mod) {
    ttnn::bind_function<"recv_async", "ttnn.experimental.">(
        mod,
        R"doc(
        Performs a recv operation from a :attr:`mesh_socket`.

        Args:
            output_tensor (ttnn.Tensor): Tensor to receive the data into.
            mesh_socket (ttnn.MeshSocket): MeshSocket to receive the data from.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Returns:
            std::vector<ttnn.Tensor>: A vector containing the output tensor.

        )doc",
        &ttnn::experimental::recv_async,
        nb::arg("output_tensor"),
        nb::arg("mesh_socket"));
}

}  // namespace ttnn::operations::experimental::ccl
