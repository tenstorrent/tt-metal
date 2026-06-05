// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dummy_op_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "dummy_op.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dummy_op::detail {

void bind_dummy_op(nb::module_& mod) {
    ttnn::bind_function<"dummy_op", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Dummy op for measuring DRAM read/write round-trip cost.

            Loops num_iter times: reads every tile of input_tensor from DRAM into a
            circular buffer, then writes them back to the same DRAM addresses
            (in-place). Returns the input tensor itself.

            Args:
                input_tensor (ttnn.Tensor): TILE-layout DRAM-interleaved tensor.
                num_iter (int): Number of read+write iterations. Baked into the
                    kernel as a compile-time constant, so changing it triggers a
                    program-cache miss / recompile.
                global_semaphore (ttnn.GlobalSemaphore): Semaphore the reader
                    kernel waits on (== 0) once before its iter loop. Must be
                    created on the cores the op will run on.
                subdevice_id (ttnn.SubDeviceId, optional): Sub-device whose
                    worker cores should host the reader/writer kernels. The
                    sub-device must span exactly one Tensix row. If omitted,
                    defaults to row 0 of the full compute grid.

            Returns:
                ttnn.Tensor: The same input_tensor (in-place op).
        )doc",
        &ttnn::operations::experimental::deepseek_prefill::dummy_op::dummy_op,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("num_iter"),
        nb::arg("global_semaphore"),
        nb::arg("subdevice_id") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dummy_op::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_dummy_op(::nanobind::module_& mod) { dummy_op::detail::bind_dummy_op(mod); }

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
