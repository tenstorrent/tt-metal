// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
        Performs a buffered recv paired with buffered_send.

        Unlike recv_direct_async (which takes a single output tensor), buffered_recv takes N
        output tensors that act as a ring of receive buffers and the receive :attr:`mesh_socket`.
        Buffer availability is coordinated through an internally-allocated, zero-initialized
        persistent L1_SMALL buffer (no caller-provided global semaphore is required).

        Note:
            This op returns None. The actual buffer written by each send is selected by device-side
            ring state.

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
