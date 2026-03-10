// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "normal_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "normal.hpp"

namespace ttnn::operations::normal {
void bind_normal_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Generates a tensor with the given shape, filled with random values sampled
        from a normal (Gaussian) distribution using the Box-Muller transform.

        Args:
            shape (list[int]): A list of integers defining the shape of the output tensor.
            device (ttnn.Device | ttnn.MeshDevice): The device on which the tensor will be allocated.

        Keyword args:
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to ``ttnn.bfloat16``.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to ``ttnn.TILE_LAYOUT``.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to ``ttnn.DRAM_MEMORY_CONFIG``.
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Must be non-negative. Defaults to 1.0.
            seed (int, optional): Seed for reproducible results. If ``None``, a random seed is used.

        Returns:
            ttnn.Tensor: A tensor with the specified shape, dtype, and layout containing samples from N(mean, std^2).
        )doc";

    ttnn::bind_function<"normal">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::normal,
            nb::arg("shape"),
            nb::arg("device"),
            nb::kw_only(),
            nb::arg("dtype") = nb::cast(DataType::BFLOAT16),
            nb::arg("layout") = nb::cast(Layout::TILE),
            nb::arg("memory_config") = nb::cast(ttnn::DRAM_MEMORY_CONFIG),
            nb::arg("mean") = 0.0f,
            nb::arg("std") = 1.0f,
            nb::arg("seed") = nb::none()));
}
}  // namespace ttnn::operations::normal
