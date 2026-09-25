// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gated_delta_rule_nanobind.hpp"
#include "chunk_gated_delta_rule.hpp"
#include "chunk_gated_delta_rule_config.hpp"
#include "device/chunk_gdn_phased.hpp"
#include "device/chunk_gdn_fused.hpp"

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/device.hpp"

#include <fmt/format.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <string>

namespace ttnn::operations::transformer {

namespace {

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
    const std::optional<ttnn::transformer::ChunkGdnProgramConfig>& program_config,
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
        program_config,
        memory_config,
        gdn_default_compute_kernel_config(q.device()->arch(), compute_kernel_config),
        eye,
        tril,
        ones,
        masks);
}

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
    uint32_t Hk,
    bool prep_serial) {
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
        Hk,
        prep_serial);
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
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    bool use_mcast,
    bool force_serial) {
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
        gdn_default_compute_kernel_config(v_beta.device()->arch(), compute_kernel_config),
        use_mcast,
        force_serial);
}

std::string py_bool(bool b) { return b ? "True" : "False"; }
std::string py_opt(const std::optional<uint32_t>& v) { return v.has_value() ? std::to_string(*v) : "None"; }
std::string py_opt(const std::optional<bool>& v) { return v.has_value() ? py_bool(*v) : "None"; }

}  // namespace

void bind_chunk_gated_delta_rule(nb::module_& mod) {
    using ttnn::transformer::ChunkGdnFusedProgramConfig;
    using ttnn::transformer::ChunkGdnMonoProgramConfig;
    using ttnn::transformer::ChunkGdnPhasedProgramConfig;

    nb::class_<ChunkGdnMonoProgramConfig>(
        mod,
        "ChunkGdnMonoProgramConfig",
        R"doc(chunk_gated_delta_rule on a simple single-kernel path: one core per head for every chunk.
        The slowest path, kept as the benchmark/debug reference; it does not accept
        flat (token-major) q/k/v.)doc")
        .def(nb::init<>())
        .def("__repr__", [](const ChunkGdnMonoProgramConfig&) { return std::string("ChunkGdnMonoProgramConfig()"); });

    nb::class_<ChunkGdnPhasedProgramConfig>(
        mod,
        "ChunkGdnPhasedProgramConfig",
        R"doc(chunk_gated_delta_rule on the two-phase path: prep phase fanned over the grid, seven fp32
        intermediates stored in DRAM, followed by the scan phase.

        Keyword Args:
            use_mcast (bool): default True. One sender core per head multicasts the scan's six shared
                inputs to that head's sibling core; False makes every core read them from DRAM.
            scan_serial (bool): default False. One core per head (NV=1, the full V) instead of the
                V-block split. Used for validation only.
            prep_serial (bool): default False. BH prep cores (one per head) instead of the whole grid.
                Validation only.)doc")
        .def(
            nb::init<bool, bool, bool>(),
            nb::kw_only(),
            nb::arg("use_mcast") = true,
            nb::arg("scan_serial") = false,
            nb::arg("prep_serial") = false)
        .def_rw("use_mcast", &ChunkGdnPhasedProgramConfig::use_mcast)
        .def_rw("scan_serial", &ChunkGdnPhasedProgramConfig::scan_serial)
        .def_rw("prep_serial", &ChunkGdnPhasedProgramConfig::prep_serial)
        .def("__repr__", [](const ChunkGdnPhasedProgramConfig& c) {
            return fmt::format(
                "ChunkGdnPhasedProgramConfig(use_mcast={}, scan_serial={}, prep_serial={})",
                py_bool(c.use_mcast),
                py_bool(c.scan_serial),
                py_bool(c.prep_serial));
        });

    nb::class_<ChunkGdnFusedProgramConfig>(
        mod,
        "ChunkGdnFusedProgramConfig",
        R"doc(chunk_gated_delta_rule on the fused path: one program in which, per head, NP producer
        cores run prep and NoC-write the seven intermediates into NV receiver cores' CBs.
        The geometry fields default to the calibrated cost model's pick for
        (grid, BH, NC, Vt) (see chunk_gdn_fused_geometry); pinning one of num_producers /
        num_receivers makes the model fill the other so the pair still fits the grid.

        Keyword Args:
            num_producers (int, optional): NP per head, clamped to the chunk count.
            num_receivers (int, optional): NV per head; must divide Vt = V / 32.
            row_local (bool, optional): core map. True: one head per row with its producers east of
                its receivers, so no two heads share a NoC link. False: row-major 1xNV receiver
                rectangles with the producers on the remaining cores. None: row-local whenever the
                geometry has such a layout.
            handoff_depth (int): default 2. Hand-off ring slots per CB (1..8): how many chunks a
                producer may run ahead of its receivers.
            unicast (bool): default True. Per-receiver unicast writes; False sends the linked
                multicast chain.
            posted (bool): default False. Posted unicast data writes with the VALID flag ordered by
                in-order delivery; requires unicast.)doc")
        .def(
            nb::init<std::optional<uint32_t>, std::optional<uint32_t>, std::optional<bool>, uint32_t, bool, bool>(),
            nb::kw_only(),
            nb::arg("num_producers") = nb::none(),
            nb::arg("num_receivers") = nb::none(),
            nb::arg("row_local") = nb::none(),
            nb::arg("handoff_depth") = 2,
            nb::arg("unicast") = true,
            nb::arg("posted") = false)
        .def_rw("num_producers", &ChunkGdnFusedProgramConfig::num_producers)
        .def_rw("num_receivers", &ChunkGdnFusedProgramConfig::num_receivers)
        .def_rw("row_local", &ChunkGdnFusedProgramConfig::row_local)
        .def_rw("handoff_depth", &ChunkGdnFusedProgramConfig::handoff_depth)
        .def_rw("unicast", &ChunkGdnFusedProgramConfig::unicast)
        .def_rw("posted", &ChunkGdnFusedProgramConfig::posted)
        .def("__repr__", [](const ChunkGdnFusedProgramConfig& c) {
            return fmt::format(
                "ChunkGdnFusedProgramConfig(num_producers={}, num_receivers={}, row_local={}, handoff_depth={}, "
                "unicast={}, posted={})",
                py_opt(c.num_producers),
                py_opt(c.num_receivers),
                py_opt(c.row_local),
                c.handoff_depth,
                py_bool(c.unicast),
                py_bool(c.posted));
        });

    // Host-side geometry oracle: what the fused op will choose for (grid, BH, NC, Vt) when the program
    // config leaves the geometry free. Pure function of its arguments — no device needed — so the
    // dispatch-table tests can run for any grid.
    mod.def(
        "chunk_gdn_fused_geometry",
        [](uint32_t grid_x,
           uint32_t grid_y,
           uint32_t BH,
           uint32_t NC,
           uint32_t Vt,
           uint32_t fixed_nv,
           uint32_t fixed_np) {
            const auto c = ttnn::prim::choose_fused_geometry(grid_x, grid_y, BH, NC, Vt, fixed_nv, fixed_np);
            return std::make_tuple(c.nv, c.np, c.placement, c.t_fused_us, c.t_phased_us, c.fused_pays);
        },
        nb::arg("grid_x"),
        nb::arg("grid_y"),
        nb::arg("BH"),
        nb::arg("NC"),
        nb::arg("Vt") = 4,
        nb::arg("fixed_nv") = 0,
        nb::arg("fixed_np") = 0,
        R"doc(Fused prep->scan geometry the op picks for (grid_x, grid_y, BH, NC, Vt) when the fused
        program config leaves it free (fixed_nv / fixed_np = a pinned num_receivers / num_producers, 0 =
        free): (nv, np, placement, T_fused_us, T_phased_us, fused_pays). nv == 0 means no fused geometry
        fits the grid.)doc");
    mod.def(
        "chunk_gdn_fused_row_local_feasible",
        &ttnn::prim::fused_row_local_feasible,
        nb::arg("grid_x"),
        nb::arg("grid_y"),
        nb::arg("BH"),
        nb::arg("NV"),
        nb::arg("NP"));
    mod.def(
        "chunk_gdn_fused_placement",
        [](uint32_t grid_x, uint32_t grid_y, uint32_t BH, uint32_t NV, uint32_t NP, uint32_t placement) {
            const auto pl = ttnn::prim::fused_placement(grid_x, grid_y, BH, NV, NP, placement);
            auto to_xy = [](const std::vector<CoreCoord>& cores) {
                std::vector<std::tuple<uint32_t, uint32_t>> xy;
                xy.reserve(cores.size());
                for (const auto& c : cores) {
                    xy.emplace_back(static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y));
                }
                return xy;
            };
            return std::make_tuple(to_xy(pl.receivers), to_xy(pl.producers));
        },
        nb::arg("grid_x"),
        nb::arg("grid_y"),
        nb::arg("BH"),
        nb::arg("NV"),
        nb::arg("NP"),
        nb::arg("placement"),
        R"doc(The fused program's core map for (grid_x, grid_y, BH, NV, NP, placement), computed by the
        same function the program factory calls: (receivers, producers), lists of logical (x, y);
        receivers[h*NV + v], producers[h*NP + j]. Raises when the layout does not fit.)doc");

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
                [B*HV, T, V] in TILE layout; otherwise token-major [B, T, HV, V] ROW_MAJOR.
            program_config (ChunkGdnFusedProgramConfig | ChunkGdnPhasedProgramConfig |
                ChunkGdnMonoProgramConfig, optional): which device implementation runs and how it is
                laid out on the chip — the alternative you pass selects the path, its fields the
                geometry (see each class). None: the fused path or the phased path depending on the cost model.
            memory_config (ttnn.MemoryConfig, optional).
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional).
            eye, tril, ones (ttnn.Tensor, optional): [1,1,C,C] fp32 TILE constant tiles (identity,
                lower-triangular ones, all-ones). Caller-supplied. Traced callers MUST pass these;
                if omitted they are built eagerly.
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
        nb::arg("program_config") = nb::none(),
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
            prep_serial (bool): default False. ChunkGdnPhasedProgramConfig.prep_serial: BH cores
                (one per head) instead of the whole grid. Measurement only.

        Returns:
            list[ttnn.Tensor]: the 7 fp32 per-chunk DRAM intermediates the scan consumes —
                [v_beta, kd, q_decay, intra, k_dec_t, dl, t_inv]; shapes/dtypes are
                ChunkGdnScanInputs, device/chunk_gdn_phased.hpp:133-142 (dl is a per-chunk
                scalar in tile position [0,0]).
        )doc";

    // Plain mod.def, without bind_function's __ttnn_operation__ marker: the two prims stay at
    // ttnn._ttnn.operations.transformer (like the geometry helpers above) instead of being registered
    // as public ttnn.transformer operations — their signatures follow the implementation.
    mod.def(
        "chunk_gdn_prep",
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
        nb::arg("Hk") = 0,
        nb::arg("prep_serial") = false,
        nb::call_guard<nb::gil_scoped_release>(),
        prep_doc);

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
            use_mcast (bool) / force_serial (bool): ChunkGdnPhasedProgramConfig.use_mcast /
                scan_serial — the shared-input multicast (default True) and the one-core-per-head
                NV=1 layout (default False; measurement only).

        Returns:
            list[ttnn.Tensor]: [o [BH, NC, C, V] fp32, final_state [BH, K, V] fp32]
                (o is fp32 — see the factory's compute_output_specs, which is authoritative).
        )doc";

    mod.def(  // private, as chunk_gdn_prep above
        "chunk_gdn_scan",
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
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("use_mcast") = true,
        nb::arg("force_serial") = false,
        nb::call_guard<nb::gil_scoped_release>(),
        scan_doc);
}

}  // namespace ttnn::operations::transformer
