// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_to_interleaved_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "sharded_to_interleaved.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_sharded_to_interleaved(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Converts a sharded tensor to an interleaved tensor (Quasar / Metal 2.0).

        Args:
            input_tensor (ttnn.Tensor): the sharded input tensor.
            memory_config (ttnn.MemoryConfig): interleaved memory configuration for the output.

        Keyword Args:
            output_dtype (ttnn.DataType, optional): data type of the output tensor. Defaults to the input dtype.

        Note:
            MX tensors (``ttnn.MX_DTYPES``: ``ttnn.mxfp8_e4m3``, ``ttnn.mxfp8_e5m2``, ``ttnn.mxfp6_e2m3``, ``ttnn.mxfp6_e3m2``, ``ttnn.mxfp4``, ``ttnn.mxint8``, ``ttnn.mxint4``, ``ttnn.mxint2``) are TILE layout only and keep their format, or convert
            to or from BFLOAT16 through ``output_dtype``.

        Returns:
            ttnn.Tensor: the interleaved output tensor.
        )doc";

    ttnn::bind_function<"sharded_to_interleaved", "ttnn.experimental.quasar.">(
        mod,
        doc,
        &ttnn::operations::experimental::quasar::sharded_to_interleaved,
        nb::arg("input_tensor").noconvert(),
        nb::arg("memory_config"),
        nb::arg("output_dtype") = nb::none());
}

}  // namespace ttnn::operations::experimental::quasar::detail
