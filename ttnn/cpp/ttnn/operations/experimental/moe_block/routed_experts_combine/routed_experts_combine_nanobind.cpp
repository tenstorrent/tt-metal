// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_experts_combine_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/export_enum.hpp"

#include "device/combine_host.hpp"
#include "routed_experts_combine.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

namespace routed_experts_combine_nanobind_detail {

nb::dict layout_dict(const CombineLayout& l) {
    nb::dict d;
    d["slot"] = l.slot;
    d["acc_fp32"] = l.acc_fp32;
    d["acc_cbs"] = l.acc_cbs;
    d["rings"] = l.rings;
    d["fits"] = l.fits;
    d["why"] = l.why;
    if (l.has_PSL) {
        d["PSL"] = l.PSL;
        d["D"] = l.D;
    }
    if (l.has_rings) {
        d["QP"] = l.QP;
        d["QL"] = l.QL;
        d["acc_used"] = l.acc_used;
        d["port_used"] = l.port_used;
    }
    return d;
}

nb::dict choice_dict(const CombineAutoChoice& c) {
    nb::dict d;
    d["acc_on_ports"] = c.acc_on_ports;
    d["n_acc"] = c.n_acc;
    d["inj_slots"] = c.inj_slots;
    d["depth"] = c.depth;
    d["local_keep"] = c.local_keep;
    d["layout"] = layout_dict(c.layout);
    return d;
}

}  // namespace routed_experts_combine_nanobind_detail

void bind_routed_experts_combine(nb::module_& mod) {
    using namespace routed_experts_combine_nanobind_detail;

    export_enum<CombineMode>(mod, "CombineMode");

    nb::class_<CombineConfig> cfg(mod, "CombineConfig", R"doc(
        The combine's op-specific configuration (the tuning switches, the tie rule and combine_snap come from
        MoEBlockConfig). n_acc / mode pin the accumulator count and mode the fused head planned; arena_bytes is the
        combine's L1 budget a core (0: the reference exchange's sizing from the largest free L1 block at the first
        call, less flat_static_bytes and the dispatch's CBs); depth is the landing pool's slots per source chip;
        piece_max the fabric piece bytes (default max(7168, 2 Hr): one piece a row).
    )doc");
    cfg.def(nb::init<>())
        .def_rw("n_acc", &CombineConfig::n_acc)
        .def_rw("mode", &CombineConfig::mode)
        .def_rw("arena_bytes", &CombineConfig::arena_bytes)
        .def_rw("flat_static_bytes", &CombineConfig::flat_static_bytes)
        .def_rw("depth", &CombineConfig::depth)
        .def_rw("piece_max", &CombineConfig::piece_max)
        .def(nb::self == nb::self);

    ttnn::bind_function<"routed_experts_combine", "ttnn.experimental.moe_block.">(
        mod,
        R"doc(
        The flat routed experts and the sparse combine in one program per chip: every expert output row goes from the
        flat expert's y writers through L1 into a fabric port's injection ring (or a local accumulator's ring), home
        over the 2D fabric (Y first, then X) into an accumulator's landing ring, and is summed there in fp32 (gate
        weight x row, snapped to multiples of combine_snap when set: bit-identical in any landing order).

        Args:
            plan: the device plan [1, 1024] uint32 (the fused head's planner).
            table: the L1 hand-off table [rows_cap / 32, 64] uint32 (the fused head's planner).
            resources: the block's MoEBlockResources (config, placement, persistent words).
            cells: the logical cells the flat plan leaves free (the combine's ports and accumulators).
            writers: the flat expert's y writers (logical), in its program's order.
            writer_words_offset: their L1 hand-off words, relative to the program's arena.
            combine: CombineConfig.

        Returns:
            y_own [1, 1, S, Hr] bf16 row-major (persistent: one buffer for every call).
        )doc",
        &routed_experts_combine,
        nb::arg("plan").noconvert(),
        nb::arg("table").noconvert(),
        nb::arg("resources"),
        nb::kw_only(),
        nb::arg("cells"),
        nb::arg("writers"),
        nb::arg("writer_words_offset"),
        nb::arg("combine") = CombineConfig{});

    // ---- host math (no device): parity with the reference, plans and reports
    mod.def(
        "combine_pieces",
        [](uint32_t W, std::optional<uint32_t> piece_max) {
            const auto p = row_pieces(W, piece_max);
            return nb::make_tuple(p.row_bytes, p.NT, p.NP, p.PD, p.PS, p.slot);
        },
        nb::arg("W"),
        nb::arg("piece_max").none() = nb::none(),
        "sparse_combine_rf._pieces: (row bytes, tiles, pieces, piece bytes, piece stride, slot bytes).");
    mod.def(
        "combine_state_layout",
        [](uint32_t N, uint32_t A) {
            const auto s = state_layout(N, A);
            return nb::make_tuple(s.CRED, s.USED, s.RET, s.LAYR, s.LAYO, s.U0, s.CNT, s.MISC, s.HDRS, s.RINGS);
        },
        nb::arg("N"),
        nb::arg("A"),
        "sparse_combine_rf._state: (CRED, USED, RET, LAYR, LAYO, U0, CNT, MISC, HDRS, RINGS).");
    mod.def(
        "combine_acc_cb_bytes",
        &acc_cb_bytes,
        nb::arg("S"),
        nb::arg("A"),
        nb::arg("NT"),
        "sparse_combine_rf._acc_cbs.");
    mod.def(
        "combine_l1_layout",
        [](uint32_t N,
           uint32_t S,
           uint32_t W,
           uint32_t A,
           uint32_t arena_bytes,
           bool acc_on_ports,
           uint32_t inj_slots,
           uint32_t depth,
           std::optional<uint32_t> piece_max,
           uint32_t local_keep) {
            return layout_dict(
                combine_l1_layout(N, S, W, A, arena_bytes, acc_on_ports, inj_slots, depth, piece_max, local_keep));
        },
        nb::arg("N"),
        nb::arg("S"),
        nb::arg("W"),
        nb::arg("A"),
        nb::kw_only(),
        nb::arg("arena_bytes"),
        nb::arg("acc_on_ports") = false,
        nb::arg("inj_slots") = 64,
        nb::arg("depth") = 8,
        nb::arg("piece_max").none() = nb::none(),
        nb::arg("local_keep") = 8,
        "sparse_combine_rf.combine_l1_layout (the same dict).");
    mod.def(
        "combine_auto",
        [](uint32_t N,
           uint32_t S,
           uint32_t W,
           uint32_t n_cells,
           uint32_t port_cells,
           uint32_t arena_bytes,
           uint32_t depth,
           std::optional<uint32_t> piece_max,
           uint32_t a_max,
           std::optional<CombineMode> mode) {
            const auto r = combine_auto(N, S, W, n_cells, port_cells, arena_bytes, depth, piece_max, a_max, mode);
            nb::dict modes;
            if (r.ports_apart) {
                modes["ports apart"] = choice_dict(*r.ports_apart);
            }
            if (r.acc_on_ports) {
                modes["acc_on_ports"] = choice_dict(*r.acc_on_ports);
            }
            nb::dict d = r.best ? choice_dict(*r.best) : nb::dict();
            d["fits"] = r.fits;
            if (!r.fits) {
                d["why"] = r.why;
            }
            d["modes"] = modes;
            return d;
        },
        nb::arg("N"),
        nb::arg("S"),
        nb::arg("W"),
        nb::arg("n_cells"),
        nb::arg("port_cells"),
        nb::kw_only(),
        nb::arg("arena_bytes"),
        nb::arg("depth") = 8,
        nb::arg("piece_max").none() = nb::none(),
        nb::arg("a_max") = kAccMax,
        nb::arg("mode").none() = nb::none(),
        "sparse_combine_rf.combine_auto (the same dict; mode a CombineMode).");
    mod.def(
        "combine_credit_policy",
        [](uint32_t N, uint32_t S, uint32_t A, uint32_t PSL, const CombineTuning& tuning, uint32_t max_rows_per_token) {
            const auto p = credit_policy(N, S, A, PSL, tuning, max_rows_per_token);
            return nb::make_tuple(p.cred_thr, p.cred_idle, p.cred_age, p.end_only ? "end" : "default");
        },
        nb::arg("N"),
        nb::arg("S"),
        nb::arg("A"),
        nb::arg("PSL"),
        nb::arg("tuning"),
        nb::arg("max_rows_per_token") = 0,
        "The combine's credit policy: (CRED_THR, CRED_IDLE, CRED_AGE, 'default' | 'end').");
    mod.def("combine_fused_arena_reserve", &fused_arena_reserve, nb::arg("flat_static_bytes"));
    mod.def(
        "combine_exchange_arena_reserve",
        &exchange_arena_reserve,
        nb::arg("flat_static_bytes"),
        nb::arg("dispatch_static_bytes"),
        nb::arg("manifest_bytes"),
        nb::arg("dispatch_in_program"),
        "k3 exchange_arena_reserve from its parts (bytes a bank kept below the exchange's arena).");
    mod.def(
        "combine_exchange_arena_bytes",
        &exchange_arena_bytes,
        nb::arg("largest_free_bytes"),
        nb::arg("reserve"),
        "The exchange's arena bytes a core from the largest free L1 block and the reserve (k3).");
    mod.def(
        "dispatch_static_cb_bytes",
        &dispatch_static_cb_bytes,
        nb::arg("N"),
        nb::arg("S"),
        nb::arg("row_width"),
        nb::kw_only(),
        nb::arg("group") = 8,
        nb::arg("manifest_batch") = 16,
        nb::arg("dtype_bytes") = 2,
        "sparse_all_gather_2d.static_cb_bytes.");
    mod.def(
        "combine_plan_reference",
        [](const std::vector<uint16_t>& ids,
           const std::vector<uint16_t>& wbits,
           uint32_t me,
           uint32_t R,
           uint32_t C,
           uint32_t S,
           uint32_t K,
           uint32_t EPC,
           uint32_t NG,
           uint32_t A,
           uint32_t L,
           uint32_t P,
           const std::array<uint8_t, 8>& lut,
           uint32_t rows_cap,
           bool wrap_x,
           bool wrap_y,
           bool token_major,
           FabricTieRule tie) {
            const auto t = plan_reference(
                ids, wbits, me, R, C, S, K, EPC, NG, A, L, P, lut, rows_cap, wrap_x, wrap_y, token_major, tie);
            nb::dict d;
            d["counts"] = t.counts;
            d["regions"] = t.regions;
            d["token_index"] = t.token_index;
            d["table"] = t.table;
            d["plan"] = t.plan;
            return d;
        },
        nb::arg("ids"),
        nb::arg("wbits"),
        nb::arg("me"),
        nb::arg("R"),
        nb::arg("C"),
        nb::arg("S"),
        nb::arg("K"),
        nb::arg("EPC"),
        nb::arg("NG"),
        nb::arg("A"),
        nb::arg("L"),
        nb::arg("P"),
        nb::arg("lut"),
        nb::arg("rows_cap"),
        nb::kw_only(),
        nb::arg("wrap_x") = false,
        nb::arg("wrap_y") = false,
        nb::arg("token_major") = false,
        nb::arg("tie") = FabricTieRule::NorthEast,
        "combine_plan.plan_reference (ids / wbits flattened [N, S, K]): the planner's tables for chip me.");
}

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
