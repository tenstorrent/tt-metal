// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_bh_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "sdpa_bh.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::experimental::llama31_8b {

void bind_sdpa_bh(nb::module_& llama31_8b) {
    const auto* doc = R"doc(
        Causal scaled dot product attention for the Llama-3.1-8B prefill shapes on Blackhole.

        Output is bit-identical to ``ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, ...)``
        with the same scale, program config and compute kernel config. Call ``sdpa_bh_supported`` first and use
        ``ttnn.transformer.scaled_dot_product_attention`` when it returns False.

        Args:
            input_tensor_q (ttnn.Tensor): [1, 32, S, 128] bfloat8_b, TILE, DRAM interleaved.
            input_tensor_k (ttnn.Tensor): [1, 8, S, 128] bfloat8_b, TILE, DRAM interleaved.
            input_tensor_v (ttnn.Tensor): [1, 8, S, 128] bfloat8_b, TILE, DRAM interleaved.
            program_config (ttnn.SDPAProgramConfig): 11x10 grid; q / k chunk 64, S = 1024.

        Keyword Args:
            scale (float, optional): defaults to 1 / sqrt(128).
            memory_config (ttnn.MemoryConfig, optional): output memory config.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): HiFi4, fp32 dest accumulation, no approx.
            output_concat_heads (bool, optional): write [1, 1, S, 32 * 128] (what nlp_concat_heads returns). Defaults to False.

        Returns:
            ttnn.Tensor: [1, 32, S, 128], or [1, 1, S, 4096] with output_concat_heads.
        )doc";

    ttnn::bind_function<"sdpa_bh", "ttnn.experimental.llama31_8b.">(
        llama31_8b,
        doc,
        &sdpa_bh,
        nb::arg("input_tensor_q"),
        nb::arg("input_tensor_k"),
        nb::arg("input_tensor_v"),
        nb::arg("program_config"),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output_concat_heads") = false);

    const auto* doc_supported = R"doc(
        True when ``sdpa_bh`` accepts these tensors and configs (Blackhole, the Llama-3.1-8B prefill shapes).
        )doc";

    ttnn::bind_function<"sdpa_bh_supported", "ttnn.experimental.llama31_8b.">(
        llama31_8b,
        doc_supported,
        nb::overload_cast<
            const ttnn::Tensor&,
            const ttnn::Tensor&,
            const ttnn::Tensor&,
            const std::optional<ttnn::operations::transformer::SDPAProgramConfig>&,
            std::optional<DeviceComputeKernelConfig>>(&sdpa_bh_supported),
        nb::arg("input_tensor_q"),
        nb::arg("input_tensor_k"),
        nb::arg("input_tensor_v"),
        nb::arg("program_config"),
        nb::kw_only(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::experimental::llama31_8b
