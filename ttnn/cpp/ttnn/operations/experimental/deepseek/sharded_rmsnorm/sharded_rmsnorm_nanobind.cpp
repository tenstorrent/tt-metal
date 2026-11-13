// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_rmsnorm_nanobind.hpp"

#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/sharded_rmsnorm/sharded_rmsnorm.hpp"

namespace ttnn::operations::experimental::deepseek::sharded_rmsnorm::detail {

void bind_sharded_rmsnorm(nb::module_& mod) {
    ttnn::bind_function<"sharded_rmsnorm", "ttnn.experimental.deepseek.">(
        mod,
        R"doc(
        Fast single-row RMSNorm on a WIDTH_SHARDED L1 tensor ``[1, 1, 1, D]``.

        Computes ``x * rsqrt(mean(x^2) + epsilon) * weight`` directly on ``input_tensor``'s
        shard cores (with cross-core scalar reduction through core 0), avoiding resharding and
        interleaved round-trips. Optionally outputs ``ROW_MAJOR`` directly in L1 ``WIDTH_SHARDED``
        or ``DRAM`` interleaved.

        Args:
            input_tensor (ttnn.Tensor): ``[1, 1, 1, D]`` BFLOAT16 L1 WIDTH_SHARDED tensor
                (``TILE`` 32x32 with shard ``[32, W_c]`` or ``ROW_MAJOR`` with shard ``[1, W_c]``,
                ``W_c % 32 == 0``).
            weight (ttnn.Tensor): ``[1, 1, 1, D]`` BFLOAT16 ``TILE`` 32x32 INTERLEAVED gamma weight.
            epsilon (float): RMSNorm epsilon (default ``1e-6``).
            memory_config (Optional[ttnn.MemoryConfig]): Output memory config. Defaults to
                ``WIDTH_SHARDED`` L1 on ``input_tensor``'s grid with shard height 32 (``TILE``) or
                1 (``ROW_MAJOR``). Also accepts ``INTERLEAVED`` (e.g. ``DRAM_MEMORY_CONFIG``).
            output_layout (Optional[ttnn.Layout]): Output layout (``TILE`` or ``ROW_MAJOR``).
                Defaults to ``input_tensor.layout``.

        Returns:
            ttnn.Tensor: ``[1, 1, 1, D]`` normalized BFLOAT16 tensor.
        )doc",
        &ttnn::experimental::deepseek::sharded_rmsnorm,
        nb::arg("input_tensor"),
        nb::arg("weight"),
        nb::arg("epsilon") = 1e-6f,
        nb::arg("memory_config") = std::nullopt,
        nb::arg("output_layout") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek::sharded_rmsnorm::detail

namespace ttnn::operations::experimental::deepseek::detail {

void bind_sharded_rmsnorm(::nanobind::module_& mod) { sharded_rmsnorm::detail::bind_sharded_rmsnorm(mod); }

}  // namespace ttnn::operations::experimental::deepseek::detail
