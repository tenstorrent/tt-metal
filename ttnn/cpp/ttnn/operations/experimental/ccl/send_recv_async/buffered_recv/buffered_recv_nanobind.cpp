// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "buffered_recv_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/buffered_recv/buffered_recv.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_buffered_recv(nb::module_& mod) {
    ttnn::bind_function<"buffered_recv", "ttnn.experimental.">(
        mod,
        R"doc(
        Receives data sent by buffered_send.

        Unlike recv_direct_async, which takes a single output tensor, this takes N tensors forming a
        ring of receive buffers. All of them must share a shape, dtype, layout and memory config.

        Note:
            Which buffer a given send lands in is chosen by device-side ring state, not by the
            caller, so callers must track the rotation themselves.

        Args:
            output_tensors (List[ttnn.Tensor]): Tensors to receive the data into.
            mesh_socket (ttnn.MeshSocket): MeshSocket to receive the data from.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        )doc",
        &ttnn::experimental::buffered_recv,
        nb::arg("output_tensors"),
        nb::arg("mesh_socket"));
}

}  // namespace ttnn::operations::experimental::ccl
