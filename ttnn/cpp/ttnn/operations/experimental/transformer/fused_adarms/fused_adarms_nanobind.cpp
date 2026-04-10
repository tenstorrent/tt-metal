// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "fused_adarms_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "fused_adarms.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::transformer {

void bind_fused_adarms(nb::module_& mod) {
    const char* doc = R"doc(
        fused_adarms(input_tensor, dense_weight, dense_bias, cond, epsilon=1e-6, *, memory_config=None)

        Fused adaptive RMSNorm for Pi0.5 transformer blocks.

        Computes the linear projection of the conditioning vector into
        (scale, shift, gate), then applies RMSNorm with scale as weight
        and shift as bias. Returns a tuple (normed, gate).

        Equivalent to:
            modulation = ttnn.linear(cond, dense_weight, bias=dense_bias)
            scale, shift, gate = ttnn.chunk(modulation, 3, dim=-1)
            normed = ttnn.rms_norm(input_tensor, weight=scale, bias=shift, epsilon=epsilon)
            return normed, gate

        Note: dense_bias should have the +1 offset pre-baked into the scale
        portion (first hidden_dim elements).

        Args:
            input_tensor (ttnn.Tensor): Input tensor [batch, seq_len, hidden_dim].
            dense_weight (ttnn.Tensor): Projection weight [cond_dim, hidden_dim * 3].
            dense_bias (ttnn.Tensor): Projection bias [1, hidden_dim * 3].
            cond (ttnn.Tensor): Conditioning vector [batch, 1, cond_dim] or [batch, cond_dim].
            epsilon (float): Small constant for numerical stability. Default: 1e-6.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Output memory config.

        Returns:
            Tuple[ttnn.Tensor, ttnn.Tensor]: (normed, gate)
    )doc";

    ttnn::bind_function<"fused_adarms", "ttnn.experimental.">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::experimental::fused_adarms,
            nb::arg("input_tensor"),
            nb::arg("dense_weight"),
            nb::arg("dense_bias"),
            nb::arg("cond"),
            nb::arg("epsilon") = 1e-6f,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

}  // namespace ttnn::operations::experimental::transformer
