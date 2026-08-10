// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "qr_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include "qr.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::qr {

void bind_qr_operation(nb::module_& mod) {
    std::string doc =
        R"doc(
        Computes the reduced QR factorization of a rank-2 fp32 matrix with both
        dimensions at most 32.

        Returns (Q, R) with Q (m x k) and R (k x n), k = min(m, n), using the
        LAPACK sign convention (sign(0) = 1).

        Args:
            input (ttnn.Tensor): The input matrix, rank-2, Float32, TILE layout.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration
                for the output tensors. Defaults to the input's memory config.

        Returns:
            Tuple[ttnn.Tensor, ttnn.Tensor]: (Q, R).

        )doc";

    ttnn::bind_function<"qr">(
        mod,
        doc.c_str(),
        &ttnn::qr,
        nb::arg("input"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}
}  // namespace ttnn::operations::qr
