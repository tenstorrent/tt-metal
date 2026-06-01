// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_decode_full_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deltanet/deltanet_decode_full.hpp"

namespace ttnn::operations::experimental::deltanet::detail {

void bind_deltanet_decode_full(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Fully fused DeltaNet decode step: conv1d + recurrence + norm + gate on device.

        Takes raw linear projection outputs and performs the entire DeltaNet
        decode step on device with zero host transfers.

        Args:
            qkv_proj (ttnn.Tensor): [1,1,1,conv_dim] raw QKV projection
            z_proj (ttnn.Tensor): [1,1,1,H*Dv] gating projection
            b_proj (ttnn.Tensor): [1,1,1,H] beta projection
            a_proj (ttnn.Tensor): [1,1,1,H] decay projection
            conv_state (ttnn.Tensor): [1,1,conv_dim,K] sliding window state
            recurrent_state (ttnn.Tensor): [1,H,Dk,Dv] recurrent state
            conv1d_weight (ttnn.Tensor): [1,1,conv_dim,K] convolution weights
            a_log (ttnn.Tensor): [1,1,1,H] decay base (log-space)
            dt_bias (ttnn.Tensor): [1,1,1,H] timestep bias
            norm_weight (ttnn.Tensor): [1,1,1,Dv] RMSNorm weight

        Keyword Args:
            num_heads (int): Number of value/output heads.
            num_k_heads (int): Number of key heads (before expansion).
            k_head_dim (int): Key head dimension.
            v_head_dim (int): Value head dimension.
            conv_dim (int): Total convolution dimension.
            conv_kernel_size (int): Convolution kernel width.
            head_expand_ratio (int): num_heads / num_k_heads.
            memory_config (ttnn.MemoryConfig, optional): output memory config.

        Returns:
            list[ttnn.Tensor]: [output, new_recurrent_state, new_conv_state]
        )doc";

    ttnn::bind_function<"deltanet_decode_full", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::deltanet_decode_full,
        nb::arg("qkv_proj"),
        nb::arg("z_proj"),
        nb::arg("b_proj"),
        nb::arg("a_proj"),
        nb::arg("conv_state"),
        nb::arg("recurrent_state"),
        nb::arg("conv1d_weight"),
        nb::arg("a_log"),
        nb::arg("dt_bias"),
        nb::arg("norm_weight"),
        nb::kw_only(),
        nb::arg("num_heads"),
        nb::arg("num_k_heads"),
        nb::arg("k_head_dim"),
        nb::arg("v_head_dim"),
        nb::arg("conv_dim"),
        nb::arg("conv_kernel_size"),
        nb::arg("head_expand_ratio"),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::deltanet::detail
