// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_decode_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deltanet/deltanet_decode.hpp"

namespace ttnn::operations::experimental::deltanet::detail {

void bind_deltanet_decode(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Fused DeltaNet decode step: single-token recurrent state update on device.

        Computes per-head:
          S_new = S * decay + outer(k, (v - k @ S) * beta)
          output = q @ S_new

        Args:
            query (ttnn.Tensor): [1, num_heads, 1, k_dim] post-L2norm query
            key (ttnn.Tensor): [1, num_heads, 1, k_dim] post-L2norm key
            value (ttnn.Tensor): [1, num_heads, 1, v_dim] value
            decay (ttnn.Tensor): [1, num_heads, 1, 1] exp(gate) scalar per head
            beta (ttnn.Tensor): [1, num_heads, 1, 1] sigmoid(beta) scalar per head
            state (ttnn.Tensor): [1, num_heads, k_dim, v_dim] recurrent state

        Keyword Args:
            num_heads (int): Number of DeltaNet heads.
            k_head_dim (int): Key head dimension (must be multiple of 32).
            v_head_dim (int): Value head dimension (must be multiple of 32).
            memory_config (ttnn.MemoryConfig, optional): output memory config.

        Returns:
            list[ttnn.Tensor]: [output, new_state]
        )doc";

    ttnn::bind_function<"deltanet_decode", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::deltanet_decode,
        nb::arg("query"),
        nb::arg("key"),
        nb::arg("value"),
        nb::arg("decay"),
        nb::arg("beta"),
        nb::arg("state"),
        nb::kw_only(),
        nb::arg("num_heads"),
        nb::arg("k_head_dim"),
        nb::arg("v_head_dim"),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::deltanet::detail
