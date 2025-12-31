// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async_nanobind.hpp"

#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "recv_async.hpp"
#include <tt-metalium/mesh_socket.hpp>

namespace nb = nanobind;

namespace ttnn::operations::experimental::ccl {

void bind_recv_async(nb::module_& mod) {
    const char* doc = R"doc(
        Performs a recv operation from a :attr:`mesh_socket`.

        Args:
            output_tensor (ttnn.Tensor): Tensor to receive the data into.
            mesh_socket (ttnn.MeshSocket): MeshSocket to receive the data from.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Returns:
            std::vector<ttnn.Tensor>: A vector containing the output tensor.

        )doc";

    mod.def("recv_async", &ttnn::experimental::recv_async, doc, nb::arg("output_tensor"), nb::arg("mesh_socket"));
}

}  // namespace ttnn::operations::experimental::ccl
