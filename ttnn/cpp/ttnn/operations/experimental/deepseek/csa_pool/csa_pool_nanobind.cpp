// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "csa_pool_nanobind.hpp"

#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/csa_pool/csa_pool.hpp"

namespace ttnn::operations::experimental::deepseek::csa_pool::detail {

void bind_csa_pool_window(nb::module_& mod) {
    ttnn::bind_function<"csa_pool_window", "ttnn.experimental.deepseek.">(
        mod,
        R"doc(
        Fused CSA compressor pool (deepseek_v4_flash ``DeepSeekV4CSACompressor._pool_window``).

        Takes ROW_MAJOR WIDTH_SHARDED inputs and produces the softmax-gated Ca/Cb
        combination as a single device op::

            prev_g = prev_gate + position_bias
            cur_g  = win_gate  + position_bias
            new_kv   = concat(prev_kv[..., :Dh], win_kv[..., Dh:], dim=window)
            new_gate = concat(prev_g[..., :Dh],  cur_g[..., Dh:],  dim=window)
            out = sum(softmax(new_gate, dim=window) * new_kv, dim=window)

        Window tensors are ``[1, 1, users * compress_rate, 2*Dh]`` width-sharded over
        ``2*Dh``. Ca lives on the first half of that grid and Cb on the second.
        ``position_bias`` is the same grid with height ``compress_rate``.

        Args:
            prev_kv (ttnn.Tensor): previous window KV, ROW_MAJOR WIDTH_SHARDED.
            prev_gate (ttnn.Tensor): previous window gate, same layout as ``prev_kv``.
            win_kv (ttnn.Tensor): current window KV, same layout as ``prev_kv``.
            win_gate (ttnn.Tensor): current window gate, same layout as ``prev_kv``.
            position_bias (ttnn.Tensor): ``[1, 1, compress_rate, 2*Dh]`` ROW_MAJOR
                WIDTH_SHARDED on the same core grid.

        Keyword Args:
            memory_config (Optional[ttnn.MemoryConfig]): output memory config. Defaults to
                WIDTH_SHARDED L1 on the Ca (first-half) cores, shape ``[1, 1, users, Dh]``.
            compute_kernel_config (Optional[ttnn.DeviceComputeKernelConfig]): compute
                settings. Defaults to HiFi4 / fp32 dest acc.

        Returns:
            ttnn.Tensor: ``[1, 1, users, Dh]`` ROW_MAJOR WIDTH_SHARDED pooled entry.
        )doc",
        &ttnn::experimental::deepseek::csa_pool_window,
        nb::arg("prev_kv"),
        nb::arg("prev_gate"),
        nb::arg("win_kv"),
        nb::arg("win_gate"),
        nb::arg("position_bias"),
        nb::kw_only(),
        nb::arg("memory_config") = std::nullopt,
        nb::arg("compute_kernel_config") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek::csa_pool::detail

namespace ttnn::operations::experimental::deepseek::detail {

void bind_csa_pool_window(::nanobind::module_& mod) { csa_pool::detail::bind_csa_pool_window(mod); }

}  // namespace ttnn::operations::experimental::deepseek::detail
