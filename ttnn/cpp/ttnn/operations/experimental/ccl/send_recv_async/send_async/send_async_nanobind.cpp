// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async_nanobind.hpp"

#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "send_async.hpp"
#include <tt-metalium/mesh_socket.hpp>

namespace nb = nanobind;

namespace ttnn::operations::experimental::ccl {

void bind_send_async(nb::module_& mod) {
    const char* doc = R"doc(
        Performs a send operation on multi-device :attr:`input_tensor` to a :attr:`mesh_socket`.

        Args:
            input_tensor (ttnn.Tensor): device tensor.
            mesh_socket (ttnn.MeshSocket): MeshSocket to send the tensor to.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Returns:
            std::vector<ttnn.Tensor>: an empty vector.

        )doc";

    mod.def("send_async", &ttnn::experimental::send_async, doc, nb::arg("input_tensor"), nb::arg("mesh_socket"));
}

}  // namespace ttnn::operations::experimental::ccl
