// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_expert_ffn_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/quasar/routed_expert_ffn/routed_expert_ffn.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar::detail {

void bind_routed_expert_ffn(nb::module_& mod) {
    ttnn::bind_function<"routed_expert_ffn", "ttnn.experimental.quasar.">(
        mod,
        R"doc(
            One routed expert's FFN without an activation, on a single node: one reader, one compute and one
            writer thread.

            y = ((x @ w_gate) * (x @ w_up)) @ w_down

            All tensors must be bfloat16, TILE layout and DRAM interleaved.

            Args:
                x (ttnn.Tensor): tokens, shape (M, K).
                w_gate (ttnn.Tensor): gate projection, shape (K, H).
                w_up (ttnn.Tensor): up projection, shape (K, H).
                w_down (ttnn.Tensor): down projection, shape (H, K).
                memory_config (ttnn.MemoryConfig, optional): output memory configuration; must be DRAM
                    interleaved. Defaults to `x`'s.

            Returns:
                ttnn.Tensor: y, shape (M, K), bfloat16, TILE layout.
        )doc",
        ttnn::overload_t(
            &ttnn::operations::experimental::quasar::routed_expert_ffn,
            nb::arg("x"),
            nb::arg("w_gate"),
            nb::arg("w_up"),
            nb::arg("w_down"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

}  // namespace ttnn::operations::experimental::quasar::detail
