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
    ttnn::transformer::ChunkGdnWyInverse wy_inverse,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& eye,
    const std::optional<ttnn::Tensor>& tril,
    const std::optional<ttnn::Tensor>& ones,
    const std::optional<ttnn::Tensor>& masks,
    const std::optional<ttnn::Tensor>& sel,
    bool qk_prenormed,
    bool decay_sfpu,
    const std::optional<ttnn::Tensor>& final_state_output) {
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
        wy_inverse,
        memory_config,
        gdn_default_compute_kernel_config(q.device()->arch(), compute_kernel_config),
        eye,
        tril,
        ones,
        masks,
        sel,
        qk_prenormed,
        decay_sfpu,
        final_state_output);
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
    bool prep_serial,
    bool gb_flat,
    const std::optional<ttnn::Tensor>& sel,
    ttnn::transformer::ChunkGdnWyInverse wy_inverse) {
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
        prep_serial,
        gb_flat,
        sel,
        wy_inverse);
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
std::string py_opt(const std::optional<tt::tt_metal::MathFidelity>& v) {
    if (!v.has_value()) {
        return "None";
    }
    switch (*v) {
        case tt::tt_metal::MathFidelity::LoFi: return "MathFidelity.LoFi";
        case tt::tt_metal::MathFidelity::HiFi2: return "MathFidelity.HiFi2";
        case tt::tt_metal::MathFidelity::HiFi3: return "MathFidelity.HiFi3";
        case tt::tt_metal::MathFidelity::HiFi4: return "MathFidelity.HiFi4";
        default: return "MathFidelity.Invalid";
    }
}

}  // namespace

void bind_chunk_gated_delta_rule(nb::module_& mod) {
    using ttnn::transformer::ChunkGdnFusedProgramConfig;
    using ttnn::transformer::ChunkGdnMonoProgramConfig;
    using ttnn::transformer::ChunkGdnPhasedProgramConfig;
    using ttnn::transformer::ChunkGdnWyInverse;

    // The WY-inverse method is arithmetic, not topology, so it is its own kwarg rather than a
    // program-config field; see chunk_gated_delta_rule_config.hpp.
    nb::enum_<ChunkGdnWyInverse>(
        mod,
        "ChunkGdnWyInverse",
        R"doc(How chunk_gated_delta_rule computes each chunk's WY inverse T_inv = (I + N)^-1. The methods
        agree to ~1e-3 (PCC-class), so this is a numerics choice and deliberately not part of the program
        config; for a given method the fused, phased and mono paths stay bit-exact with each other.)doc")
        .value(
            "AUTO",
            ChunkGdnWyInverse::AUTO,
            "the SFPU solve wherever it is supported (Blackhole, chunk_size == 32), Horner everywhere else")
        .value(
            "HORNER",
            ChunkGdnWyInverse::HORNER,
            "quadrant-split Horner inverses on the matrix engine; every architecture and chunk size; the reference")
        .value(
            "SFPU",
            ChunkGdnWyInverse::SFPU,
            "one SFPU forward-substitution solve reading the factor as fp32 in place; Blackhole-only, chunk_size == "
            "32, and refused (not downgraded) where unsupported");

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
                in-order delivery; requires unicast.
            split_layout (bool): default False. The SPLIT core map: the first BH/2 heads put their
                receivers at the top of their own column with the producers below, the other half is
                mirrored at the bottom of the grid (those producers write on NOC_0), so the two halves'
                hand-off traffic uses disjoint column links. Used whenever it fits the grid (BH even,
                2*NV < grid.y, ...); otherwise row_local decides. Bit-exact with every other core map.
            qk_fp32_double_buffer (bool): default False. Double-buffer the producers' fp32 q/k CBs
                (qk_prenormed with FLOAT32 q/k only; no effect otherwise): +32 KB of producer CB region
                at K = 128, hides the fp32 q/k read on producer-bound geometries. Bit-exact.
            prep_math_fidelity (ttnn.MathFidelity, optional): math fidelity of the producer (prep)
                compute kernel. None: the op's compute_kernel_config fidelity (HiFi4).
            scan_math_fidelity (ttnn.MathFidelity, optional): math fidelity of the receiver (scan)
                compute kernel. None: the op's compute_kernel_config fidelity (HiFi4).
                Unlike the other fields, the two fidelities change the arithmetic. Both are hashed, so a
                change compiles a new program. The experiment env vars QWEN36_FLA_PREP_FID /
                QWEN36_FLA_SCAN_FID, when set, take precedence over these fields.)doc")
        .def(
            nb::init<
                std::optional<uint32_t>,
                std::optional<uint32_t>,
                std::optional<bool>,
                uint32_t,
                bool,
                bool,
                bool,
                bool,
                std::optional<tt::tt_metal::MathFidelity>,
                std::optional<tt::tt_metal::MathFidelity>>(),
            nb::kw_only(),
            nb::arg("num_producers") = nb::none(),
            nb::arg("num_receivers") = nb::none(),
            nb::arg("row_local") = nb::none(),
            nb::arg("handoff_depth") = 2,
            nb::arg("unicast") = true,
            nb::arg("posted") = false,
            nb::arg("split_layout") = false,
            nb::arg("qk_fp32_double_buffer") = false,
            nb::arg("prep_math_fidelity") = nb::none(),
            nb::arg("scan_math_fidelity") = nb::none())
        .def_rw("num_producers", &ChunkGdnFusedProgramConfig::num_producers)
        .def_rw("num_receivers", &ChunkGdnFusedProgramConfig::num_receivers)
        .def_rw("row_local", &ChunkGdnFusedProgramConfig::row_local)
        .def_rw("handoff_depth", &ChunkGdnFusedProgramConfig::handoff_depth)
        .def_rw("unicast", &ChunkGdnFusedProgramConfig::unicast)
        .def_rw("posted", &ChunkGdnFusedProgramConfig::posted)
        .def_rw("split_layout", &ChunkGdnFusedProgramConfig::split_layout)
        .def_rw("qk_fp32_double_buffer", &ChunkGdnFusedProgramConfig::qk_fp32_double_buffer)
        .def_rw("prep_math_fidelity", &ChunkGdnFusedProgramConfig::prep_math_fidelity)
        .def_rw("scan_math_fidelity", &ChunkGdnFusedProgramConfig::scan_math_fidelity)
        .def("__repr__", [](const ChunkGdnFusedProgramConfig& c) {
            return fmt::format(
                "ChunkGdnFusedProgramConfig(num_producers={}, num_receivers={}, row_local={}, handoff_depth={}, "
                "unicast={}, posted={}, split_layout={}, qk_fp32_double_buffer={}, prep_math_fidelity={}, "
                "scan_math_fidelity={})",
                py_opt(c.num_producers),
                py_opt(c.num_receivers),
                py_opt(c.row_local),
                c.handoff_depth,
                py_bool(c.unicast),
                py_bool(c.posted),
                py_bool(c.split_layout),
                py_bool(c.qk_fp32_double_buffer),
                py_opt(c.prep_math_fidelity),
                py_opt(c.scan_math_fidelity));
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
            wy_inverse (ttnn.ChunkGdnWyInverse): default AUTO. How each chunk's WY inverse is
                computed: HORNER on the matrix engine (every architecture; the reference), SFPU
                (one forward-substitution solve, Blackhole and chunk_size 32 only, refused where
                unsupported) or AUTO (SFPU wherever supported, else Horner). The two methods agree
                to ~1e-3; unlike program_config this changes bits, though every path stays
                bit-exact with the others for a given method.
            memory_config (ttnn.MemoryConfig, optional).
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): every path runs HiFi4 with
                fp32 destination accumulation and no approx mode; a config asking for other arithmetic
                is rejected rather than applied.
            eye, tril, ones (ttnn.Tensor, optional): [1,1,C,C] fp32 TILE constant tiles (identity,
                lower-triangular ones, all-ones). Caller-supplied. Traced callers MUST pass these;
                if omitted they are built eagerly.
            masks (ttnn.Tensor, optional): [1,1,32,96] fp32 TILE quadrant masks; supplied with eye/
                tril/ones.
            sel (ttnn.Tensor, optional): [1,1,32,32*HV] fp32 TILE one-hot head selector (tile h
                picks head h's column). Passing it enables gb_flat (Option B: g/beta read straight
                from [B,T,HV], skipping the permute+reshape prep; fused path only — pass it only
                with a ChunkGdnFusedProgramConfig or with None when the cost model picks fused;
                phased/mono reject it). Omit it to keep the head-split g/beta path. Build it once
                on the model/layer (device-resident before trace capture, same as eye/tril/ones/masks).
            qk_prenormed (bool): default False. q/k arrive already L2-normalized per head over K (q also
                multiplied by scale; k / sqrt(sum k^2 + 1e-6), the in-kernel norm's formula) as flat
                [B, T, H*K] BFLOAT16 or FLOAT32 TILE tensors: the in-kernel norm is skipped and FLOAT32 q/k
                are read as fp32 (no bf16 cast). Flat q/k, chunk_size 32 and the fused path only.
            decay_sfpu (bool): default False. Compute each chunk's decay chain (cumulative decay, its
                exponentials, the decay mask and dl*I) in two fp32 SFPU passes instead of ~13 single-tile
                FPU ops: faster and more accurate (fp32 instead of tf32 operands), so it changes bits.
                Fused path and chunk_size 32 only.
            final_state_output (ttnn.Tensor, optional): a pre-allocated final-state tensor, FLOAT32
                TILE interleaved, [B, HV, K, V] or [B*HV, K, V], any buffer type. The kernel writes the
                final state straight into it (no new state tensor is allocated) and the op returns it
                as final_state. It may be the initial_state tensor itself (in-place state update, for
                a persistent traced state buffer). Needs output_final_state=True; fused and phased
                paths only (mono rejects it).

        Returns:
            tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
                o [B, T, HV, V] (or [B*HV, T, V] if output_head_major),
                final_state [B, HV, K, V] (if output_final_state; final_state_output itself when given).
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
        nb::arg("wy_inverse") = ttnn::transformer::ChunkGdnWyInverse::AUTO,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("eye") = nb::none(),
        nb::arg("tril") = nb::none(),
        nb::arg("ones") = nb::none(),
        nb::arg("masks") = nb::none(),
        nb::arg("sel") = nb::none(),
        nb::arg("qk_prenormed") = false,
        nb::arg("decay_sfpu") = false,
        nb::arg("final_state_output") = nb::none());

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
                by the same helper as the public op's default; other arithmetic is rejected.
            v_flat (bool) / HV (int): OPT-A flat token-major v, chunk_gdn_phased.hpp:34-40.
            qk_norm (bool) / scale (float): OPT-B in-kernel q/k L2 norm with scale folded into
                q's norm, chunk_gdn_phased.hpp:45-48.
            qk_flat (bool) / Hk (int): OPT-A flat token-major q/k, chunk_gdn_phased.hpp:41-44.
            prep_serial (bool): default False. ChunkGdnPhasedProgramConfig.prep_serial: BH cores
                (one per head) instead of the whole grid. Measurement only.
            gb_flat (bool) / sel (ttnn.Tensor, optional): Option B flat token-major g/beta — g/beta
                are then the raw [B,T,HV] fp32 tensor (HV<=32) instead of [BH,NC,C,1], and `sel`
                ([1,1,32,32*HV] fp32 TILE one-hot head selector) is REQUIRED. The phased prep
                prim rejects it (fused path only); see chunk_gdn_phased.hpp's ChunkGdnPrepParams::gb_flat.
            wy_inverse (ttnn.ChunkGdnWyInverse): default AUTO — the WY-inverse method, as on the
                public op (HORNER / SFPU / AUTO).

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
        nb::arg("gb_flat") = false,
        nb::arg("sel") = nb::none(),
        nb::arg("wy_inverse") = ttnn::transformer::ChunkGdnWyInverse::AUTO,
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
            dl (ttnn.Tensor):      [BH, NC, 32, 32] fp32: dl*I, the per-chunk decay exp(g_sum) on the diagonal
            t_inv (ttnn.Tensor):   [BH, NC, C, C] fp32 (WY inverse)

        Keyword Args:
            initial_state (ttnn.Tensor, optional): [BH, K, V] fp32; absent means zeros.
            chunk_size (int): default 32 (C).
            output_final_state (bool): default True; when False the final_state slot's
                contents are unspecified.
            memory_config (ttnn.MemoryConfig, optional): default DRAM interleaved (the public
                op's default).
            compute_kernel_config (optional): default HiFi4 + fp32 dest acc, no approx — built
                by the same helper as the public op's default; other arithmetic is rejected.
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
