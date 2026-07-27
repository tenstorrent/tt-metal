// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gated_delta_rule_nanobind.hpp"
#include "chunk_gated_delta_rule.hpp"

#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

namespace ttnn::operations::transformer {

void bind_chunk_gated_delta_rule(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Standalone chunked Gated Delta Rule forward (flash-linear-attention algorithm).

        Args:
            q (ttnn.Tensor):    [B, T, H,  K]
            k (ttnn.Tensor):    [B, T, H,  K]
            v (ttnn.Tensor):    [B, T, HV, V]
            g (ttnn.Tensor):    [B, T, HV]   log-space decay
            beta (ttnn.Tensor): [B, T, HV]

        Keyword Args:
            scale (float, optional): defaults to K**-0.5.
            initial_state (ttnn.Tensor, optional): [B, HV, K, V].
            output_final_state (bool): default False.
            chunk_size (int): default 64.
            use_qk_l2norm (bool): default False.
            output_head_major (bool): default False. When True, o is returned head-major as
                [B*HV, T, V] in TILE layout (skips the token<->head permute round-trip);
                otherwise token-major [B, T, HV, V] ROW_MAJOR.
            memory_config (ttnn.MemoryConfig, optional).
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional).
            eye, tril, ones (ttnn.Tensor, optional): [1,1,C,C] fp32 TILE constant tiles (identity,
                lower-triangular ones, all-ones). Caller-supplied so they are device-resident before
                trace capture and their lifetime is device-scoped. Traced callers MUST pass these
                (an internal build does a host upload, illegal under trace); if omitted they are
                built eagerly.
            masks (ttnn.Tensor, optional): [1,1,32,96] fp32 TILE quadrant masks; supplied with eye/
                tril/ones.

        Returns:
            tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
                o [B, T, HV, V] (or [B*HV, T, V] if output_head_major),
                final_state [B, HV, K, V] (if output_final_state).
        )doc";

    ttnn::bind_function<"chunk_gated_delta_rule", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::chunk_gated_delta_rule,
        nb::arg("q").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("g").noconvert(),
        nb::arg("beta").noconvert(),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("initial_state") = nb::none(),
        nb::arg("output_final_state") = false,
        nb::arg("chunk_size") = 64,
        nb::arg("use_qk_l2norm") = false,
        nb::arg("output_head_major") = false,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("eye") = nb::none(),
        nb::arg("tril") = nb::none(),
        nb::arg("ones") = nb::none(),
        nb::arg("masks") = nb::none());

    ttnn::bind_function<"chunk_kda", "ttnn.transformer.">(
        mod,
        R"doc(
        Chunk-parallel Kimi Delta Attention recurrence with per-key vector decay.

        q/k must be L2-normalized. Shapes: q/k/g [B,T,H,K], v [B,T,H,V],
        with rank-3 flat [B,T,H*D] q/k/v/g accepted for tile-aligned sequences;
        beta [B,T,H], initial_state [B,H,K,V]. chunk_size is currently 32.
        summary_group_chunks counts 32-token chunks in each local affine-summary group.
        Returns token-major output [B,T,H,V], or TILE [B*H,T,V] when output_head_major=True,
        and an optional final state.
        )doc",
        &ttnn::transformer::chunk_kda,
        nb::arg("q").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("g").noconvert(),
        nb::arg("beta").noconvert(),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("initial_state") = nb::none(),
        nb::arg("output_final_state") = false,
        nb::arg("output_head_major") = false,
        nb::arg("chunk_size") = 32,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("eye") = nb::none(),
        nb::arg("tril") = nb::none(),
        nb::arg("ones") = nb::none(),
        nb::arg("masks") = nb::none(),
        nb::arg("rms_gate") = nb::none(),
        nb::arg("rms_weight") = nb::none(),
        nb::arg("rms_epsilon") = 1e-5f,
        nb::arg("summary_group_chunks") = 8,
        nb::arg("sequence_parallel_axis") = nb::none(),
        nb::arg("affine_identity") = nb::none(),
        nb::arg("affine_zero") = nb::none());

    ttnn::bind_function<"kda_distributed_affine_prefix", "ttnn.transformer.">(
        mod,
        R"doc(
        Compose one affine KDA partition summary per SP rank with a logarithmic
        causal prefix. Returns each rank entry state and the global final state
        replicated over the SP mesh axis.
        )doc",
        &ttnn::transformer::kda_distributed_affine_prefix,
        nb::arg("transform_a").noconvert(),
        nb::arg("transform_b").noconvert(),
        nb::arg("initial_state").noconvert(),
        nb::arg("identity_a").noconvert(),
        nb::arg("zero_b").noconvert(),
        nb::kw_only(),
        nb::arg("sequence_parallel_axis"),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());

    ttnn::bind_function<"kda_convolution_halo", "ttnn.transformer.">(
        mod,
        "Exchange causal-convolution carry between SP neighbors and replicate the final carry.",
        &ttnn::transformer::kda_convolution_halo,
        nb::arg("projected_qkv").noconvert(),
        nb::arg("initial_carry").noconvert(),
        nb::kw_only(),
        nb::arg("sequence_parallel_axis"),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());

    ttnn::bind_function<"kda_gated_rms_norm", "ttnn.transformer.">(
        mod,
        "Fused per-head RMSNorm and sigmoid gate for tile-aligned KDA prefill.",
        &ttnn::transformer::kda_gated_rms_norm,
        nb::arg("input").noconvert(),
        nb::arg("gate").noconvert(),
        nb::arg("weight").noconvert(),
        nb::arg("num_heads"),
        nb::kw_only(),
        nb::arg("epsilon") = 1e-5f,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());

    ttnn::bind_function<"kda_causal_conv1d_split", "ttnn.transformer.">(
        mod,
        "Four-tap KDA convolution with direct tiled Q/K/V outputs.",
        &ttnn::transformer::kda_causal_conv1d_split,
        nb::arg("input").noconvert(),
        nb::arg("state").noconvert(),
        nb::arg("tap0").noconvert(),
        nb::arg("tap1").noconvert(),
        nb::arg("tap2").noconvert(),
        nb::arg("tap3").noconvert(),
        nb::arg("q_width"),
        nb::arg("k_width"),
        nb::arg("v_width"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::transformer
