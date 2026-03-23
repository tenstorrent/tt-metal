// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_nanobind.hpp"

#include <cstdint>

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deltanet/deltanet.hpp"

namespace ttnn::operations::experimental::deltanet::detail {

void bind_deltanet_recurrence(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Performs fused DeltaNet recurrence on device.

        Computes the delta rule recurrence for all heads in a single kernel launch:
          1. state *= decay
          2. kv_mem = k @ state
          3. delta = (v - kv_mem) * beta
          4. state += k^T @ delta
          5. output = q @ state

        Args:
            conv_out (ttnn.Tensor): Convolution output (1,1,B_pad,conv_dim).
            b_proj (ttnn.Tensor): Beta projection (1,1,B_pad,num_v_heads).
            a_proj (ttnn.Tensor): Alpha projection (1,1,B_pad,num_v_heads).
            z_proj (ttnn.Tensor): Z gate projection (1,1,B_pad,value_dim).
            dt_bias (ttnn.Tensor): DT bias (1,1,1,num_v_heads).
            A_exp (ttnn.Tensor): A exponential (1,1,1,num_v_heads).
            norm_weight (ttnn.Tensor): RMSNorm weight (1,1,1,head_v_dim).
            state (ttnn.Tensor): Recurrent state (1,num_v_heads,head_k_dim,head_v_dim). Modified in-place.

        Keyword Args:
            num_heads (int): Number of value heads (48).
            head_k_dim (int): Key dimension per head (128).
            head_v_dim (int): Value dimension per head (128).
            num_k_heads (int): Number of key heads (16).
            gqa_ratio (int): GQA ratio (3).
            scale (float): Scale factor (1/sqrt(head_k_dim)).
            norm_eps (float): RMSNorm epsilon (1e-6).

        Returns:
            ttnn.Tensor: Output tensor (1,1,B_pad,value_dim).
        )doc";

    ttnn::bind_function<"deltanet_recurrence", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::deltanet_recurrence,
        nb::arg("conv_out"),
        nb::arg("b_proj"),
        nb::arg("a_proj"),
        nb::arg("z_proj"),
        nb::arg("dt_bias"),
        nb::arg("A_exp"),
        nb::arg("norm_weight"),
        nb::arg("state"),
        nb::kw_only(),
        nb::arg("num_heads"),
        nb::arg("head_k_dim"),
        nb::arg("head_v_dim"),
        nb::arg("num_k_heads"),
        nb::arg("gqa_ratio"),
        nb::arg("scale"),
        nb::arg("norm_eps"));
}

}  // namespace ttnn::operations::experimental::deltanet::detail
