// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sp_eq_mul_mask_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "sp_eq_mul_mask.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::ssm::detail {

namespace nb = nanobind;

void bind_sp_eq_mul_mask(nb::module_& mod) {
    const auto* const doc = R"doc(
        Fused eq + mask: returns A where A == B, zero elsewhere.

        Args:
            a (ttnn.Tensor): first tilized bfloat16 tensor (interleaved).
            b (ttnn.Tensor): second tilized bfloat16 tensor, same shape as a.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): output memory config; defaults to a's.
            dtype (ttnn.DataType, optional): output dtype; defaults to a's.

        Returns:
            ttnn.Tensor: tile-elementwise A * (A == B).
    )doc";

    ttnn::bind_function<"sp_eq_mul_mask", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::sp_eq_mul_mask,
        nb::arg("a"),
        nb::arg("b"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("dtype") = nb::none());
}

}  // namespace ttnn::operations::experimental::ssm::detail
