// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gpt_oss_swiglu_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "gpt_oss_swiglu.hpp"

namespace ttnn::operations::experimental::gpt_oss_swiglu {

ttnn::Tensor gpt_oss_swiglu_func(
    const ttnn::Tensor& gate_tensor,
    const ttnn::Tensor& up_tensor,
    float alpha,
    float clamp_limit,
    const std::optional<ttnn::MemoryConfig>& output_memory_config) {
    return ttnn::experimental::gpt_oss_swiglu(gate_tensor, up_tensor, alpha, clamp_limit, output_memory_config);
}

}  // namespace ttnn::operations::experimental::gpt_oss_swiglu

namespace ttnn::operations::experimental::gpt_oss_swiglu::detail {

void bind_gpt_oss_swiglu(nb::module_& mod) {
    ttnn::bind_function<"gpt_oss_swiglu", "ttnn.experimental.">(
        mod,
        R"doc(
        Fused clamped SwiGLU activation for GPT-OSS MoE.

        Computes:
          gate_clamped = clamp(gate, max=clamp_limit)
          up_clamped   = clamp(up, min=-clamp_limit, max=clamp_limit)
          result       = (up_clamped + 1) * gate_clamped * sigmoid(alpha * gate_clamped)

        Both inputs and output are L1 BLOCK_SHARDED on the same core grid; the
        compute kernel reads/writes the sharded buffers directly via globally
        allocated CBs (no reader/writer dataflow kernels). Single SFPU pass per
        tile pair makes this ~7x cheaper than the eltwise composition variant.

        Args:
            gate_tensor: [..., M, N] bf16 BLOCK_SHARDED L1 tile-layout, gate output of W1 matmul.
            up_tensor:   [..., M, N] bf16 BLOCK_SHARDED L1 tile-layout, up output of W3 matmul.
                         Must have identical shape and shard spec to gate_tensor.
            alpha:       sigmoid scale factor (default 1.702 for GPT-OSS).
            clamp_limit: clamp limit (default 7.0 for GPT-OSS).
            output_memory_config: defaults to gate_tensor's memory_config (same shard layout).
        )doc",
        &ttnn::operations::experimental::gpt_oss_swiglu::gpt_oss_swiglu_func,
        nb::arg("gate_tensor"),
        nb::arg("up_tensor"),
        nb::kw_only(),
        nb::arg("alpha") = 1.702f,
        nb::arg("clamp_limit") = 7.0f,
        nb::arg("output_memory_config") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::gpt_oss_swiglu::detail

namespace ttnn::operations::experimental::detail {

void bind_gpt_oss_swiglu(::nanobind::module_& mod) { gpt_oss_swiglu::detail::bind_gpt_oss_swiglu(mod); }

}  // namespace ttnn::operations::experimental::detail
