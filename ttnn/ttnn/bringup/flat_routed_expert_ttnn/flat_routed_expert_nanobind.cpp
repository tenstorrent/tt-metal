// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_nanobind.hpp"

#include <map>
#include <set>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/global_semaphore.hpp>

#include "flat_routed_expert.hpp"
#include "device/flat_drisc.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::bringup::detail {

namespace nb = nanobind;
namespace fre = ttnn::operations::bringup::flat_routed_expert;

namespace {
nb::list cores(const std::vector<tt::tt_metal::CoreCoord>& cs) {
    nb::list l;
    for (const auto& c : cs) {
        l.append(nb::make_tuple(c.x, c.y));
    }
    return l;
}

// The plan's layout facts the weight preparation and the done words need (see flat_routed_expert_plan.hpp).
nb::dict plan_dict(
    tt::tt_metal::distributed::MeshDevice* device,
    uint32_t hidden,
    uint32_t intermediate,
    uint32_t experts_per_chip,
    uint32_t num_global_experts,
    uint32_t max_tokens_per_expert,
    bool weights_bf8,
    uint32_t pin,
    bool x_bf16,
    bool h_bf16,
    const fre::FlatLayoutOptions& layout) {
    const auto p = fre::flat_routed_expert_plan(
        device,
        fre::FlatRoutedExpertConfig{
            .hidden = hidden,
            .intermediate = intermediate,
            .experts_per_chip = experts_per_chip,
            .num_global_experts = num_global_experts,
            .max_tokens = (max_tokens_per_expert + 31) / 32 * 32,
            .weights_bf8 = weights_bf8,
            .activation = 0,
            .pin = pin,
            .x_bf16 = x_bf16,
            .h_bf16 = h_bf16,
            .layout = layout});
    nb::dict d;
    d["nsg"] = p->nsg;
    d["n_rd"] = p->readers.size();
    d["n_rd_sg"] = p->n_rd_sg;
    d["np"] = p->np;
    d["g"] = p->g;
    d["mt"] = p->mt;
    d["rg"] = p->rg;
    d["uneven"] = p->uneven;
    d["hg"] = p->hg;
    d["nu"] = p->nu;
    d["rgs"] = p->rgs;  // (uneven split: pair-sets per reader / first pair-set; empty when even)
    d["ps0"] = p->ps0;
    d["nk_gu"] = p->nk_gu;
    d["kblk"] = 8;
    d["nd"] = p->down.size();
    d["pcds"] = p->pcds;
    d["col0s"] = p->col0s;
    d["rdown"] = p->rdown;
    d["n_rdn"] = p->n_rdn;
    d["pcd_r"] = p->pcd_r;
    d["kd_r"] = p->kd_r;
    d["nblk_r"] = p->nblk_r;
    d["rem_cols"] = p->rem_cols;
    d["banks"] = p->banks;
    d["dw_rot"] = p->dw_rot;
    d["wd_rot_tiles"] = p->w_tile ? p->wd_rot_region / p->w_tile : 0;
    d["x_slots"] = p->x_slots;
    d["hbuf"] = p->hbuf;
    d["hu"] = p->hu;
    d["gu_rp"] = p->gu_rp;
    d["gu_l1acc"] = p->gu_l1acc;
    d["arena_tiles"] = p->arena_tiles;
    // (per-role arena offsets, bytes: gate/up x ring at x_off .. h_local at hl_off; down h at h_off, out at o_off;
    // Tensix readers' rings end at rd_off; relays' staging ends at ti_off)
    d["role_offsets"] = std::map<std::string, uint32_t>{
        {"x_off", p->x_off},
        {"p_off", p->p_off},
        {"hl_off", p->hl_off},
        {"rd_off", p->rd_off},
        {"h_off", p->h_off},
        {"o_off", p->o_off},
        {"ti_off", p->ti_off}};
    d["coords"] = cores(p->coords);
    d["readers"] = cores(p->readers);
    d["gu"] = cores(p->gu);
    d["relays"] = cores(p->relays);
    d["down"] = cores(p->down);
    d["gu_idle"] = cores(p->gu_idle);
    d["arena_cores"] = cores(p->arena_cores());
    // DRISC: which reader cells still run Tensix readers (the others are free unless they became down cores) and the
    // launcher (posts the DRISC jobs when no Tensix reader is left to)
    // the per-call schedule's parameters (flat_schedule_defines without the pinning ones, which follow the call's
    // pin): another op rebuilding a chip's schedule from counts (se_dyn_from_counts) must use the same
    d["sched_rps"] = p->mt * 32;
    d["sched_max_e"] = p->E <= 16 ? 16u : (p->E <= 32 ? 32u : 64u);
    d["sched_gu_nreg"] = p->nk_gu ? p->ring_g / p->nk_gu : 0u;
    d["drisc"] = p->drisc;
    d["dual"] = p->dual;
    {
        std::vector<tt::tt_metal::CoreCoord> trd;
        for (uint32_t r = 0; r < p->readers.size(); ++r) {
            if (p->tensix_reader(r)) {
                trd.push_back(p->readers[r]);
            }
        }
        if (p->drisc && !p->dual) {
            trd.push_back(p->launcher);
        }
        d["tensix_readers"] = cores(trd);
    }
    {  // the wiring, for NoC models: reader tails (reader, chain tail), down chains (down -> successor)
        nb::list rdn, succ;
        for (const auto& [r, t] : p->rdn) {
            rdn.append(nb::make_tuple(r, t));
        }
        for (const auto& [a, b] : p->d_succ) {
            succ.append(nb::make_tuple(a, b));
        }
        d["rdn"] = rdn;
        d["d_succ"] = succ;
        d["col0s"] = p->col0s;
    }
    {  // the y writers in the program factory's order (a writer's index in the combine's L1 hand-off): the reader
       // tails (by pair-set count, then non-DRISC / DRISC slots), then the down cores grouped by column width
        std::vector<tt::tt_metal::CoreCoord> yw;
        const auto rg_at = [&](uint32_t r) { return p->rgs.empty() ? p->rg : p->rgs[r]; };
        if (p->rdown) {
            std::set<uint32_t> rgk;
            for (const auto& [r, t] : p->rdn) {
                rgk.insert(rg_at(r));
            }
            for (const uint32_t k : rgk) {
                for (const bool dual_set : {false, true}) {
                    for (const auto& [r, t] : p->rdn) {
                        if ((p->dual && p->drisc_slot(r)) == dual_set && rg_at(r) == k) {
                            yw.push_back(p->readers[r]);
                        }
                    }
                }
            }
        }
        std::map<uint32_t, std::vector<uint32_t>> dgroups;
        for (uint32_t i = 0; i < p->down.size(); ++i) {
            dgroups[p->pcds[i]].push_back(i);
        }
        for (const auto& [pw, ds] : dgroups) {
            for (const uint32_t i : ds) {
                yw.push_back(p->down[i]);
            }
        }
        d["y_writers"] = cores(yw);
    }
    return d;
}
}  // namespace

void bind_flat_routed_expert(nb::module_& mod) {
    // the layout / program choices (FlatLayoutOptions: typed, in the plan key and the program hashes); presets in
    // ttnn/ttnn/bringup/flat_routed_expert_ttnn/flat_layouts.py
    nb::class_<fre::FlatLayoutOptions>(mod, "FlatLayoutOptions")
        .def(nb::init<>())
        .def_rw("drisc", &fre::FlatLayoutOptions::drisc)
        .def_rw("drisc_dual", &fre::FlatLayoutOptions::drisc_dual)
        .def_rw("drisc_in_overlap", &fre::FlatLayoutOptions::drisc_in_overlap)
        .def_rw("drisc_pure", &fre::FlatLayoutOptions::drisc_pure)
        .def_rw("max_down", &fre::FlatLayoutOptions::max_down)
        .def_rw("x_helpers", &fre::FlatLayoutOptions::x_helpers)
        .def_rw("half_width", &fre::FlatLayoutOptions::half_width)
        .def_rw("hg_units", &fre::FlatLayoutOptions::hg_units)
        .def_rw("single_subgrid", &fre::FlatLayoutOptions::single_subgrid)
        .def_rw("row0", &fre::FlatLayoutOptions::row0)
        .def_rw("row1", &fre::FlatLayoutOptions::row1)
        .def_rw("x_path", &fre::FlatLayoutOptions::x_path)
        .def_rw("x_snakes", &fre::FlatLayoutOptions::x_snakes)
        .def_rw("x_pieces", &fre::FlatLayoutOptions::x_pieces)
        .def_rw("x_fwd_ncrisc", &fre::FlatLayoutOptions::x_fwd_ncrisc)
        .def_rw("x_fwd_noc", &fre::FlatLayoutOptions::x_fwd_noc)
        .def_rw("x_fwd_depth", &fre::FlatLayoutOptions::x_fwd_depth)
        .def_rw("x_relay_depth", &fre::FlatLayoutOptions::x_relay_depth)
        .def_rw("x_once", &fre::FlatLayoutOptions::x_once)
        .def_rw("dw_batch", &fre::FlatLayoutOptions::dw_batch)
        .def_rw("pin_ratio", &fre::FlatLayoutOptions::pin_ratio)
        .def("key", &fre::FlatLayoutOptions::key)
        .def("__repr__", [](const fre::FlatLayoutOptions& o) { return "FlatLayoutOptions(" + o.str() + ")"; })
        .def("__eq__", [](const fre::FlatLayoutOptions& a, const fre::FlatLayoutOptions& b) { return a == b; });
    ttnn::bind_function<"flat_routed_expert", "ttnn.bringup.">(
        mod,
        R"doc(
        Flat spatially pipelined routed experts (Blackhole): every local expert's SwiGLU FFN in ONE program that
        streams each expert's weights once through a spatial pipeline. Reads the row-major bf16 dispatch buffer and
        the routing's counts / regions rows on device (no tilize, no host sync).

        Args:
            dispatched_buffer (ttnn.Tensor): [rows, H] bf16 ROW_MAJOR, DRAM interleaved.
            expert_token_counts (ttnn.Tensor): [1, num_global_experts] uint32 ROW_MAJOR.
            expert_region_offsets (ttnn.Tensor): [1, num_global_experts] uint32 ROW_MAJOR.
            global_expert_ids (ttnn.Tensor): [experts_per_chip] uint32 ROW_MAJOR (per device).
            gate_up_weights, down_weights (ttnn.Tensor): the plan's bank layout (bfp4 / bfp8).
            reader_down_weights (ttnn.Tensor, optional): when the plan has reader tails (``rdown``).
            done_words (ttnn.Tensor): persistent zeroed L1 tensor on the plan's ``coords`` cores.
            intermediate (int): the expert intermediate size (per device).
            max_tokens_per_expert (int): the dispatch capacity per expert (>= 256).

        Keyword Args:
            activation (int): 0 SiLU, 1 SwiGluOai, 2 SituGlu, 3 ClampedSiluGlu, 4 GeluTanh.
            pin (int): pin the largest expert's weights in chunks of >= pin sub-blocks (0: off).
            token_index (ttnn.Tensor, optional): [1, rows] uint32 ROW_MAJOR DRAM. Indexed mode: row r of the flat
                (region) space reads x row token_index[r] (x is then e.g. the all-gathered tokens, not a dispatch
                buffer); y has token_index's rows.
            x_pages_per_row (int): x stores each token row as this many consecutive pages ([rows * P, H / P]
                row-major): P = 8 at H 4096 puts a 1 KB piece of every row in each of the 8 DRAM banks, so the reads
                stay bank-balanced whatever rows the routing picks.

            y_row_major (bool): y as row-major bf16 [rows, H] (one page per row), pack-untilized on the down cores,
                instead of bfp8 tiles.
            down_fp32 (bool): accumulate the down projection in fp32 DEST (full-sync DST) instead of bf16 DEST (whose
                rounding inflates y by ~3-4% with bfp4 weights at K 2048); ~2% more kernel time.
            pack_stochastic_rounding (bool): stochastic rounding in the compute kernels' packers (x tilize, h, y).
                The default packer rounds bf16 -> bfp8 ties away from zero, a +0.28% magnitude bias per pack on
                bf16-valued data that adds up to ~+1.25% on y; unbiased with this, at slightly more noise.
            x_bf16 (bool): x as bf16 tiles (the relays tilize to bf16; else bfp8). Must match the plan the weights were
                laid out with (flat_routed_expert_plan(..., x_bf16)).
            h_bf16 (bool): h (gate/up -> down) as bf16 tiles (else bfp8); as x_bf16.
            y_bf16 (bool): y as bf16 tiles (else bfp8; tiled y only): no bfp8 rounding of the output.
            math_fidelity (int): the expert matmuls' fidelity (0 LoFi: exact for bfp4 weights; 2 HiFi2 for bfp8).
            layout (FlatLayoutOptions): the layout / program choices (presets: flat_layouts.py); must match the
                plan the weights were laid out with (flat_routed_expert_plan(..., layout)).

        Returns:
            ttnn.Tensor: y [rows, H] bfp8 TILE, or bf16 ROW_MAJOR with y_row_major (rows: token_index's length in
            indexed mode), written at the active experts' rows.
        )doc",
        &fre::flat_routed_expert,
        nb::arg("dispatched_buffer").noconvert(),
        nb::arg("expert_token_counts").noconvert(),
        nb::arg("expert_region_offsets").noconvert(),
        nb::arg("global_expert_ids").noconvert(),
        nb::arg("gate_up_weights").noconvert(),
        nb::arg("down_weights").noconvert(),
        nb::arg("reader_down_weights") = nb::none(),
        nb::arg("done_words").noconvert(),
        nb::arg("intermediate"),
        nb::arg("max_tokens_per_expert"),
        nb::kw_only(),
        nb::arg("activation") = 0,
        nb::arg("pin") = 1,
        nb::arg("token_index") = nb::none(),
        nb::arg("x_pages_per_row") = 1,
        nb::arg("y_row_major") = false,
        nb::arg("down_fp32") = false,
        nb::arg("pack_stochastic_rounding") = false,
        nb::arg("x_bf16") = false,
        nb::arg("h_bf16") = false,
        nb::arg("y_bf16") = false,
        nb::arg("math_fidelity") = 0,
        nb::arg("layout") = fre::FlatLayoutOptions{},
        nb::arg("output") = nb::none());

    ttnn::bind_function<"flat_combine_overlap", "ttnn.bringup.">(
        mod,
        R"doc(
        The flat routed expert overlapped with combine_fabric2d in one program per chip (initial version): combine on
        grid rows 0-1 takes each local expert as soon as every flat y writer reports its rows landed. The flat expert's
        arguments as for flat_routed_expert (planned below rows 0-1: layout rows 2..9), combine's as for
        hybrid_routed_expert_moe's overlap. Returns combine's output [1, 1, seq_len_per_chip, num_experts_per_tok, H]
        bf16 ROW_MAJOR; y_out (optional, [rows, H]) receives the flat expert's output: bf16 ROW_MAJOR with y_row_major
        (default: combine's readers read its rows directly, no untilizer cores; plan with layout rows 1..9), else bfp8
        TILE (combine untilizes it on rows 0-1; layout rows 2..9).
        )doc",
        &fre::flat_combine_overlap,
        nb::arg("dispatched_buffer").noconvert(),
        nb::arg("expert_token_counts").noconvert(),
        nb::arg("expert_region_offsets").noconvert(),
        nb::arg("global_expert_ids").noconvert(),
        nb::arg("gate_up_weights").noconvert(),
        nb::arg("down_weights").noconvert(),
        nb::arg("reader_down_weights") = nb::none(),
        nb::arg("done_words").noconvert(),
        nb::arg("intermediate"),
        nb::arg("max_tokens_per_expert"),
        nb::arg("dispatched_metadata").noconvert(),
        nb::arg("expert_offsets").noconvert(),
        nb::arg("replicated_global_expert_idx_table").noconvert(),
        nb::kw_only(),
        nb::arg("num_experts_per_tok"),
        nb::arg("seq_len_per_chip"),
        nb::arg("combine_axis") = 0,
        nb::arg("combine_num_links") = 2,
        nb::arg("fwd_arrived_semaphore"),
        nb::arg("final_arrived_semaphore"),
        nb::arg("expert_go_semaphore"),
        nb::arg("activation") = 0,
        nb::arg("pin") = 1,
        nb::arg("down_fp32") = false,
        nb::arg("x_bf16") = false,
        nb::arg("h_bf16") = false,
        nb::arg("y_row_major") = true,
        nb::arg("y_out") = nb::none(),
        nb::arg("layout") = fre::FlatLayoutOptions{});
    ttnn::bind_function<"flat_routed_expert_with_program", "ttnn.bringup.">(
        mod,
        R"doc(
        The flat routed expert with another op's programs appended, one program per chip: the flat expert's arguments
        as for flat_routed_expert, plus the caller's persistent arena (over the flat expert's cores and the other op's
        cells) and y ([rows, H] bf16 ROW_MAJOR, DRAM), the other op's MeshProgramDescriptor (extra_key: its cache key),
        and where the y writers report each local expert (counts[e] += 1 per writer and call at counts_addr on the
        logical collector core; writers: the expected writer count). Returns y.
        )doc",
        &fre::flat_routed_expert_with_program,
        nb::arg("dispatched_buffer").noconvert(),
        nb::arg("expert_token_counts").noconvert(),
        nb::arg("expert_region_offsets").noconvert(),
        nb::arg("global_expert_ids").noconvert(),
        nb::arg("gate_up_weights").noconvert(),
        nb::arg("down_weights").noconvert(),
        nb::arg("reader_down_weights") = nb::none(),
        nb::arg("done_words").noconvert(),
        nb::arg("intermediate"),
        nb::arg("max_tokens_per_expert"),
        nb::kw_only(),
        nb::arg("arena").noconvert(),
        nb::arg("y").noconvert(),
        nb::arg("extra"),
        nb::arg("extra_key"),
        nb::arg("collector"),
        nb::arg("counts_addr"),
        nb::arg("writers"),
        nb::arg("activation") = 0,
        nb::arg("pin") = 1,
        nb::arg("token_index") = nb::none(),
        nb::arg("down_fp32") = false,
        nb::arg("x_bf16") = false,
        nb::arg("h_bf16") = false,
        nb::arg("l1_args") = std::vector<std::vector<uint32_t>>{},
        nb::arg("relay_words") = nb::none(),
        nb::arg("y_l1") = nb::none(),
        nb::arg("layout") = fre::FlatLayoutOptions{});
    mod.def(
        "flat_routed_expert_plan",
        &plan_dict,
        nb::arg("device"),
        nb::arg("hidden"),
        nb::arg("intermediate"),
        nb::arg("experts_per_chip"),
        nb::arg("num_global_experts"),
        nb::arg("max_tokens_per_expert"),
        nb::arg("weights_bf8") = false,
        nb::arg("pin") = 1,
        nb::arg("x_bf16") = false,
        nb::arg("h_bf16") = false,
        nb::arg("layout") = fre::FlatLayoutOptions{},
        "The flat_routed_expert layout plan (weight bank layout, coordinator cores for the done words).");
    mod.def(
        "flat_drisc_stop",
        [](tt::tt_metal::distributed::MeshDevice* device) {
            ttnn::operations::bringup::flat_routed_expert::flat_drisc_stop(device);
        },
        nb::arg("device"),
        "Stops the flat expert's persistent DRISC gate/up senders on this mesh device (layout.drisc); call before "
        "closing it.");
    mod.def(
        "flat_drisc_dump",
        [](tt::tt_metal::distributed::MeshDevice* device) {
            ttnn::operations::bringup::flat_routed_expert::flat_drisc_dump(device);
        },
        nb::arg("device"),
        "Logs the flat expert's DRISC senders' words (debug).");
}

}  // namespace ttnn::operations::bringup::detail
