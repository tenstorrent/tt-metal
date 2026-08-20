// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gated_delta_rule_nanobind.hpp"
#include "chunk_gated_delta_rule.hpp"
#include "device/chunk_gdn_phased.hpp"

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/device.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace ttnn::operations::transformer {

namespace {

// The ONE default-construction site for the GDN compute config, shared by all three bindings
// (public op + the two phased prims) so the defaults cannot drift. Must stay identical to the
// default the C++ op builds for C++ callers at chunk_gated_delta_rule.cpp:267-273:
// HiFi4, fp32 dest acc, no approx, no L1 acc.
ttnn::DeviceComputeKernelConfig gdn_default_compute_kernel_config(
    tt::ARCH arch, const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::init_device_compute_kernel_config(
        arch,
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
}

// Public-op launcher: pre-resolves the compute config through the shared helper above, then
// forwards. Behavior-identical to binding the op directly (init_device_compute_kernel_config
// returns a provided config unchanged, so the op's internal default site is bypassed with the
// same value it would have built).
std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> chunk_gated_delta_rule_launch(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    std::optional<float> scale,
    const std::optional<ttnn::Tensor>& initial_state,
    bool output_final_state,
    uint32_t chunk_size,
    bool use_qk_l2norm,
    bool output_head_major,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& eye,
    const std::optional<ttnn::Tensor>& tril,
    const std::optional<ttnn::Tensor>& ones,
    const std::optional<ttnn::Tensor>& masks) {
    return ttnn::transformer::chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        chunk_size,
        use_qk_l2norm,
        output_head_major,
        memory_config,
        gdn_default_compute_kernel_config(q.device()->arch(), compute_kernel_config),
        eye,
        tril,
        ones,
        masks);
}

// Prim launchers: apply the public op's defaults (DRAM interleaved, shared compute config)
// so a part test that omits the kwargs exercises exactly the configs the op dispatches.
std::vector<ttnn::Tensor> chunk_gdn_prep_launch(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& eye,
    const ttnn::Tensor& tril,
    const ttnn::Tensor& ones,
    const ttnn::Tensor& masks,
    uint32_t chunk_size,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    bool v_flat,
    uint32_t HV,
    bool qk_norm,
    float scale,
    bool qk_flat,
    uint32_t Hk) {
    return ttnn::prim::chunk_gdn_prep(
        q,
        k,
        v,
        g,
        beta,
        eye,
        tril,
        ones,
        masks,
        chunk_size,
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        gdn_default_compute_kernel_config(q.device()->arch(), compute_kernel_config),
        v_flat,
        HV,
        qk_norm,
        scale,
        qk_flat,
        Hk);
}

std::vector<ttnn::Tensor> chunk_gdn_scan_launch(
    const ttnn::Tensor& v_beta,
    const ttnn::Tensor& kd,
    const ttnn::Tensor& q_decay,
    const ttnn::Tensor& intra,
    const ttnn::Tensor& k_dec_t,
    const ttnn::Tensor& dl,
    const ttnn::Tensor& t_inv,
    const std::optional<ttnn::Tensor>& initial_state,
    uint32_t chunk_size,
    bool output_final_state,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::chunk_gdn_scan(
        v_beta,
        kd,
        q_decay,
        intra,
        k_dec_t,
        dl,
        t_inv,
        initial_state,
        chunk_size,
        output_final_state,
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        gdn_default_compute_kernel_config(v_beta.device()->arch(), compute_kernel_config));
}

}  // namespace

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
        &chunk_gated_delta_rule_launch,
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

    const auto* prep_doc =
        R"doc(
        Phased-GDN PREP prim — testing/debug surface for the phased path's first stage
        (the public op composes prep -> DRAM hand-off -> scan; this exposes prep alone).

        All state-independent per-(head,chunk) work of chunk_gated_delta_rule, fanned across
        cores. Inputs are HEAD-MAJOR per-chunk tensors [BH, NC, C, *]; exact shapes/dtypes are
        ChunkGdnPrepInputs, device/chunk_gdn_phased.hpp:53-63.

        Args:
            q (ttnn.Tensor):     [BH, NC, C, K] bf16
            k (ttnn.Tensor):     [BH, NC, C, K] bf16
            v (ttnn.Tensor):     [BH, NC, C, V] bf16 (or FLAT [B, T, HV*V] when v_flat)
            g (ttnn.Tensor):     [BH, NC, C, 1] fp32 column (log-space decay)
            beta (ttnn.Tensor):  [BH, NC, C, 1] fp32 column
            eye (ttnn.Tensor):   [1,1,C,C] fp32 TILE identity
            tril (ttnn.Tensor):  [1,1,C,C] fp32 TILE lower-triangular ones
            ones (ttnn.Tensor):  [1,1,C,C] fp32 TILE all-ones
            masks (ttnn.Tensor): [1,1,32,96] fp32 TILE WY-inverse quadrant masks (Qtl|Qbr|Q10)

        Keyword Args:
            chunk_size (int): default 32 (C; Ct==1 is the only qk_norm-capable config).
            memory_config (ttnn.MemoryConfig, optional): default DRAM interleaved (the public
                op's default).
            compute_kernel_config (optional): default HiFi4 + fp32 dest acc, no approx — built
                by the same helper as the public op's default.
            v_flat (bool) / HV (int): OPT-A flat token-major v, chunk_gdn_phased.hpp:34-40.
            qk_norm (bool) / scale (float): OPT-B in-kernel q/k L2 norm with scale folded into
                q's norm, chunk_gdn_phased.hpp:45-48.
            qk_flat (bool) / Hk (int): OPT-A flat token-major q/k, chunk_gdn_phased.hpp:41-44.

        Returns:
            list[ttnn.Tensor]: the 7 fp32 per-chunk DRAM intermediates the scan consumes —
                [v_beta, kd, q_decay, intra, k_dec_t, dl, t_inv]; shapes/dtypes are
                ChunkGdnScanInputs, device/chunk_gdn_phased.hpp:133-142 (dl is a per-chunk
                scalar in tile position [0,0]).
        )doc";

    ttnn::bind_function<"chunk_gdn_prep", "ttnn.transformer.">(
        mod,
        prep_doc,
        &chunk_gdn_prep_launch,
        nb::arg("q").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("g").noconvert(),
        nb::arg("beta").noconvert(),
        nb::arg("eye").noconvert(),
        nb::arg("tril").noconvert(),
        nb::arg("ones").noconvert(),
        nb::arg("masks").noconvert(),
        nb::kw_only(),
        nb::arg("chunk_size") = 32,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("v_flat") = false,
        nb::arg("HV") = 0,
        nb::arg("qk_norm") = false,
        nb::arg("scale") = 1.0f,
        nb::arg("qk_flat") = false,
        nb::arg("Hk") = 0);

    const auto* scan_doc =
        R"doc(
        Phased-GDN SCAN prim — testing/debug surface for the phased path's second stage
        (sequential over chunks, carrying the recurrent state S [K,V]; parallel over heads).

        Consumes the 7 fp32 HEAD-MAJOR per-chunk intermediates chunk_gdn_prep produced; exact
        shapes/dtypes are ChunkGdnScanInputs, device/chunk_gdn_phased.hpp:133-142.

        Args:
            v_beta (ttnn.Tensor):  [BH, NC, C, V] fp32 (= v * beta)
            kd (ttnn.Tensor):      [BH, NC, C, K] fp32 (= k_beta * decay_exp)
            q_decay (ttnn.Tensor): [BH, NC, C, K] fp32
            intra (ttnn.Tensor):   [BH, NC, C, C] fp32
            k_dec_t (ttnn.Tensor): [BH, NC, K, C] fp32
            dl (ttnn.Tensor):      [BH, NC, 1, 1] fp32 (per-chunk scalar in tile [0,0])
            t_inv (ttnn.Tensor):   [BH, NC, C, C] fp32 (WY inverse)

        Keyword Args:
            initial_state (ttnn.Tensor, optional): [BH, K, V] fp32; absent means zeros.
            chunk_size (int): default 32 (C).
            output_final_state (bool): default True; when False the final_state slot's
                contents are unspecified.
            memory_config (ttnn.MemoryConfig, optional): default DRAM interleaved (the public
                op's default).
            compute_kernel_config (optional): default HiFi4 + fp32 dest acc, no approx — built
                by the same helper as the public op's default.

        Returns:
            list[ttnn.Tensor]: [o [BH, NC, C, V] fp32, final_state [BH, K, V] fp32]
                (o is fp32 — see the factory's compute_output_specs, which is authoritative).
        )doc";

    ttnn::bind_function<"chunk_gdn_scan", "ttnn.transformer.">(
        mod,
        scan_doc,
        &chunk_gdn_scan_launch,
        nb::arg("v_beta").noconvert(),
        nb::arg("kd").noconvert(),
        nb::arg("q_decay").noconvert(),
        nb::arg("intra").noconvert(),
        nb::arg("k_dec_t").noconvert(),
        nb::arg("dl").noconvert(),
        nb::arg("t_inv").noconvert(),
        nb::kw_only(),
        nb::arg("initial_state") = nb::none(),
        nb::arg("chunk_size") = 32,
        nb::arg("output_final_state") = true,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::transformer
