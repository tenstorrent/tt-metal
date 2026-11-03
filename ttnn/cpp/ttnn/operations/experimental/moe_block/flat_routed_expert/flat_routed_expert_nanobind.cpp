// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_nanobind.hpp"

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <tuple>

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/mesh_device.hpp>

#include "device/flat_program.hpp"
#include "device/flat_gu_place.hpp"
#include "device/flat_routed_expert_plan.hpp"
#include "flat_routed_expert.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

namespace nanobind_detail {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::distributed::MeshDevice;

// A plan as Python holds it (plans are shared and immutable).
struct PlanHandle {
    std::shared_ptr<const FlatRoutedExpertPlan> plan;
};

nb::list cores(const std::vector<CoreCoord>& cs) {
    nb::list l;
    for (const auto& c : cs) {
        l.append(nb::make_tuple(c.x, c.y));
    }
    return l;
}

// The plan's layout facts the weight preparation, the done words and callers' checks need.
nb::dict plan_dict(const FlatRoutedExpertPlan& p) {
    nb::dict d;
    d["n_rd"] = p.readers.size();
    d["n_rd_sg"] = p.n_rd_sg;
    d["np"] = p.np;
    d["g"] = p.g;
    d["mt"] = p.mt;
    d["rg"] = p.rg;
    d["hg"] = p.hg;
    d["nu"] = p.nu;
    d["nk_gu"] = p.nk_gu;
    d["kblk"] = kKBlk;
    d["nd"] = p.down.size();
    d["pcds"] = p.pcds;
    d["col0s"] = p.col0s;
    d["rem_cols"] = p.rem_cols;
    d["banks"] = p.banks;
    d["wd_rot_tiles"] = p.w_tile ? p.wd_rot_region / p.w_tile : 0;
    d["x_slots"] = p.x_slots;
    d["hbuf"] = p.hbuf;
    d["gu_rp"] = p.gu_rp;
    d["drisc_n"] = p.drisc_n;
    d["drisc_tiles"] = to_string(p.drisc_tiles);
    d["arena_tiles"] = p.arena_tiles;
    d["static_l1_bytes"] = p.static_l1_bytes;
    d["role_offsets"] = std::map<std::string, uint32_t>{
        {"x_off", p.x_off}, {"hl_off", p.hl_off}, {"h_off", p.h_off}, {"o_off", p.o_off}, {"ti_off", p.ti_off}};
    d["role_bytes"] =
        std::map<std::string, uint32_t>{{"gu", p.gu_bytes}, {"down", p.dn_bytes}, {"relay", p.relay_bytes}};
    d["coords"] = cores(p.coords);
    d["readers"] = cores(p.readers);
    d["launcher"] = nb::make_tuple(p.launcher.x, p.launcher.y);
    d["gu"] = cores(p.gu);
    d["relays"] = cores(p.relays);
    d["down"] = cores(p.down);
    d["gu_idle"] = cores(p.gu_idle);
    d["arena_cores"] = cores(p.arena_cores());
    d["y_writers"] = cores(flat_y_writers(p));
    // the per-call schedule's parameters (flat_schedule_defines without the pinning ones)
    d["sched_rps"] = p.mt * 32;
    d["sched_max_e"] = p.E <= 16 ? 16u : (p.E <= 32 ? 32u : 64u);
    d["sched_gu_nreg"] = p.nk_gu ? p.ring_g / p.nk_gu : 0u;
    return d;
}

}  // namespace nanobind_detail

void bind_flat_routed_expert(nb::module_& mod) {
    using namespace nanobind_detail;

    nb::enum_<FlatDriscTiles>(mod, "FlatDriscTiles", R"doc(
        Which DRISC tiles of each DRAM bank send the flat expert's gate/up weights. ``both``: the free tile and the
        NOC1-endpoint tile, every call (default). ``free_only``: the free tile sends both of its bank's slots.
        ``noc1_endpoint_only``: the NOC1-endpoint tile sends both, on NOC0 (the free tiles stay unused).
    )doc")
        .value("both", FlatDriscTiles::both)
        .value("free_only", FlatDriscTiles::free_only)
        .value("noc1_endpoint_only", FlatDriscTiles::noc1_endpoint_only);

    nb::enum_<FlatDriscProbe>(mod, "FlatDriscProbe", R"doc(
        A measurement probe of the flat expert's DRISC senders (garbage y). ``timing``: the normal sender with cycle
        counters (FlatDriscSenders.read_prof); ``dma_only``: GDDR DMA, no grant checks / weight writes;
        ``write_only``: grants and weight writes, no DMA; ``loop_only``: the loop and semaphore increments only.
    )doc")
        .value("none", FlatDriscProbe::none)
        .value("timing", FlatDriscProbe::timing)
        .value("dma_only", FlatDriscProbe::dma_only)
        .value("write_only", FlatDriscProbe::write_only)
        .value("loop_only", FlatDriscProbe::loop_only)
        .value("call_time", FlatDriscProbe::call_time);

    nb::enum_<FlatGuSource>(mod, "FlatGuSource", R"doc(
        Who reads the flat expert's gate/up weights. ``drisc``: the DRAM cores' DRISC senders (default).
        ``tensix``: each gate/up core reads its own blocks (NCRISC, NOC0). ``tensix_split``: a bank's first slot's
        gate/up cores read on NOC1 (BRISC), the second slot's on NOC0 (NCRISC).
    )doc")
        .value("drisc", FlatGuSource::drisc)
        .value("tensix", FlatGuSource::tensix)
        .value("tensix_split", FlatGuSource::tensix_split)
        .value("tensix_noc1", FlatGuSource::tensix_noc1)
        .value("placed", FlatGuSource::placed)
        .value("auto_tensix", FlatGuSource::auto_tensix)
        .value("auto_hybrid", FlatGuSource::auto_hybrid);

    nb::class_<FlatLayoutOptions>(mod, "FlatLayoutOptions", R"doc(
        The flat routed expert's layout / program choices (part of the plan key and of the program hash). Unset
        fields keep the plan's own choice.

        Keyword Args:
            x_helpers (int): x helper relays per rectangle; only 1 (-1: 1).
            pin_ratio (int): pin an expert only at this many times the next one's rows (0: always pin).
            rows_hint (int): expected mean rows per local expert (-1: unknown); <= 320 takes 64-row sub-blocks on
                plans with 128-row ones and gate/up row passes.
            x_ready (int): > 0 - the x relays wait for this count in their x-ready word before reading x (a caller
                running the dispatch in the same program); -1: x is in place at launch.
            drisc_tiles (FlatDriscTiles): the DRISC tiles a bank sends its gate/up weight slots from (both).
            drisc_ub (int): DRISC blocks per unit (-1: the sender's choice).
            drisc_dma_cap (int): DRISC DMA reads in flight a job, 1..3 (-1: 2).
            drisc_mpfe (int): the GDDR MPFE arbitration word (se_drisc.hpp SD_MPFE; 0: untouched).
            drisc_early_free (int): DRISC L1 slots refill once their writes left L1, not once acked (-1: the plan's).
            drisc_probe (FlatDriscProbe): a DRISC sender measurement probe (none).
            gu_source (FlatGuSource): who reads the gate/up weights (drisc).
            gu_read_depth (int): gu_source tensix*: weight reads in flight a gate/up core, 1..6 (-1: 2).
            gu_read_pieces (int): each gate/up weight block read as 1 / 2 / 4 pieces (-1: 1).
            gu_place (list[int]): gu_source placed: per gate/up core (x << 24 | y << 16 | bank << 8 | how), how 1
                Tensix NOC0, 2 Tensix NOC1, 3 DRISC free tile NOC1 (the bank's slot b), 4 DRISC free tile NOC0
                (slot b + banks); DRISC ones need drisc_tiles free_only.
    )doc")
        .def(
            "__init__",
            [](FlatLayoutOptions* self,
               int32_t x_helpers,
               uint32_t pin_ratio,
               int32_t rows_hint,
               int32_t x_ready,
               FlatDriscTiles drisc_tiles,
               int32_t drisc_ub,
               int32_t drisc_dma_cap,
               uint32_t drisc_mpfe,
               int32_t drisc_early_free,
               FlatDriscProbe drisc_probe,
               FlatGuSource gu_source,
               int32_t gu_read_depth,
               int32_t gu_read_pieces,
               std::vector<uint32_t> gu_place) {
                new (self) FlatLayoutOptions{
                    .x_helpers = x_helpers,
                    .pin_ratio = pin_ratio,
                    .rows_hint = rows_hint,
                    .x_ready = x_ready,
                    .drisc_tiles = drisc_tiles,
                    .drisc_ub = drisc_ub,
                    .drisc_dma_cap = drisc_dma_cap,
                    .drisc_mpfe = drisc_mpfe,
                    .drisc_early_free = drisc_early_free,
                    .drisc_probe = drisc_probe,
                    .gu_source = gu_source,
                    .gu_read_depth = gu_read_depth,
                    .gu_read_pieces = gu_read_pieces,
                    .gu_place = std::move(gu_place)};
            },
            nb::kw_only(),
            nb::arg("x_helpers") = -1,
            nb::arg("pin_ratio") = 8,
            nb::arg("rows_hint") = -1,
            nb::arg("x_ready") = -1,
            nb::arg("drisc_tiles") = FlatDriscTiles::both,
            nb::arg("drisc_ub") = -1,
            nb::arg("drisc_dma_cap") = -1,
            nb::arg("drisc_mpfe") = 0,
            nb::arg("drisc_early_free") = -1,
            nb::arg("drisc_probe") = FlatDriscProbe::none,
            nb::arg("gu_source") = FlatGuSource::drisc,
            nb::arg("gu_read_depth") = -1,
            nb::arg("gu_read_pieces") = -1,
            nb::arg("gu_place") = std::vector<uint32_t>{})
        .def_rw("x_helpers", &FlatLayoutOptions::x_helpers)
        .def_rw("pin_ratio", &FlatLayoutOptions::pin_ratio)
        .def_rw("rows_hint", &FlatLayoutOptions::rows_hint)
        .def_rw("x_ready", &FlatLayoutOptions::x_ready)
        .def_rw("drisc_tiles", &FlatLayoutOptions::drisc_tiles)
        .def_rw("drisc_ub", &FlatLayoutOptions::drisc_ub)
        .def_rw("drisc_dma_cap", &FlatLayoutOptions::drisc_dma_cap)
        .def_rw("drisc_mpfe", &FlatLayoutOptions::drisc_mpfe)
        .def_rw("drisc_early_free", &FlatLayoutOptions::drisc_early_free)
        .def_rw("drisc_probe", &FlatLayoutOptions::drisc_probe)
        .def_rw("gu_source", &FlatLayoutOptions::gu_source)
        .def_rw("gu_read_depth", &FlatLayoutOptions::gu_read_depth)
        .def_rw("gu_place", &FlatLayoutOptions::gu_place)
        .def_rw("gu_read_pieces", &FlatLayoutOptions::gu_read_pieces)
        .def("key", &FlatLayoutOptions::key)
        .def("__repr__", [](const FlatLayoutOptions& o) { return "FlatLayoutOptions(" + o.str() + ")"; })
        .def("__eq__", [](const FlatLayoutOptions& a, const FlatLayoutOptions& b) { return a == b; });

    nb::class_<PlanHandle>(mod, "FlatRoutedExpertPlan", R"doc(
        The flat routed expert's plan for a mesh device and shape: role cores, work split, arena layout, weight bank
        layout. ``as_dict()`` gives its layout facts (what the weight preparation needs).
    )doc")
        .def("as_dict", [](const PlanHandle& h) { return plan_dict(*h.plan); })
        .def_prop_ro("drisc_n", [](const PlanHandle& h) { return h.plan->drisc_n; })
        .def_prop_ro("drisc_tiles", [](const PlanHandle& h) { return h.plan->drisc_tiles; })
        .def_prop_ro("drisc_probe", [](const PlanHandle& h) { return h.plan->drisc_probe; })
        .def_prop_ro("arena_tiles", [](const PlanHandle& h) { return h.plan->arena_tiles; });

    mod.def(
        "flat_routed_expert_plan",
        [](MeshDevice& mesh_device,
           uint32_t hidden,
           uint32_t intermediate,
           uint32_t experts_per_chip,
           uint32_t num_global_experts,
           uint32_t max_tokens_per_expert,
           bool weights_bf8,
           uint32_t pin,
           const FlatLayoutOptions& layout) {
            return PlanHandle{flat_routed_expert_plan(
                mesh_device,
                make_flat_routed_expert_config(
                    hidden,
                    intermediate,
                    experts_per_chip,
                    num_global_experts,
                    max_tokens_per_expert,
                    weights_bf8,
                    pin,
                    layout))};
        },
        nb::arg("mesh_device"),
        nb::arg("hidden"),
        nb::arg("intermediate"),
        nb::arg("experts_per_chip"),
        nb::arg("num_global_experts"),
        nb::arg("max_tokens_per_expert"),
        nb::kw_only(),
        nb::arg("weights_bf8") = false,
        nb::arg("pin") = 1,
        nb::arg("layout") = FlatLayoutOptions{},
        R"doc(
        The flat routed expert's plan (cached per mesh device and shape) for hidden H, intermediate I (per device),
        experts_per_chip local experts, num_global_experts (the counts' length) and the per-expert capacity.
        )doc");

    mod.def(
        "create_flat_routed_expert_done_words",
        [](MeshDevice& mesh_device, const PlanHandle& plan, std::optional<tt::tt_metal::BufferType> buffer_type) {
            return create_flat_routed_expert_done_words(
                mesh_device, *plan.plan, buffer_type.value_or(tt::tt_metal::BufferType::L1));
        },
        nb::arg("mesh_device"),
        nb::arg("plan"),
        nb::arg("buffer_type") = nb::none(),
        R"doc(
        The down coordinators' done words for ``plan``: a zeroed persistent uint32 tensor sharded over the plan's
        coordinator cores in L1 (or L1_SMALL). Keep it for every call of the op; the op leaves it zero.
        )doc");

    mod.def(
        "flat_drisc_topology",
        [](MeshDevice& mesh_device, uint32_t n_slots, FlatDriscTiles drisc_tiles) {
            const FlatDriscTopology t = FlatDriscTopology::of(mesh_device, drisc_tiles);
            nb::dict d;
            d["base"] = t.base;
            d["size"] = t.size;
            d["banks"] = t.banks;
            nb::list slots;
            for (uint32_t r = 0; r < n_slots; ++r) {
                const auto [ac, aj] = t.alt(r);
                const auto [sp, snoc] = t.sender(r);
                nb::dict s;
                s["xy"] = t.virtual_xy(t.logical(r));
                s["job"] = FlatDriscTopology::job(r, t.banks);
                s["alt_xy"] = aj != ~0u ? t.virtual_xy(ac) : 0u;
                s["alt_job"] = aj != ~0u ? aj : 0u;
                s["sender_noc0"] = nb::make_tuple(sp.x, sp.y);
                s["sender_noc"] = snoc;
                slots.append(s);
            }
            d["slots"] = slots;
            nb::list banks;  // every DRAM_TILE of each bank: NOC0 grid position, NoC address, endpoint mask
            for (uint32_t b = 0; b < t.banks; ++b) {
                nb::list tl;
                for (const auto& c : t.cores.at(b)) {
                    nb::dict cd;
                    cd["noc0"] = nb::make_tuple(c.noc0_core.x, c.noc0_core.y);
                    cd["xy"] = static_cast<uint32_t>((c.virtual_core.x << 16) | c.virtual_core.y);
                    cd["endpoint_mask"] = static_cast<uint32_t>(c.endpoint_noc_mask);
                    cd["usable"] = c.usable;
                    tl.append(cd);
                }
                banks.append(tl);
            }
            d["banks"] = banks;
            const auto g = mesh_device.compute_with_storage_grid_size();
            nb::dict wk;  // logical worker (x, y) -> NOC0 grid position
            for (uint32_t y = 0; y < g.y; ++y) {
                for (uint32_t x = 0; x < g.x; ++x) {
                    const auto n0 = flat_worker_noc0(mesh_device, CoreCoord(x, y));
                    wk[nb::make_tuple(x, y)] = nb::make_tuple(n0.x, n0.y);
                }
            }
            d["workers_noc0"] = wk;
            return d;
        },
        nb::arg("mesh_device"),
        nb::arg("n_slots"),
        nb::arg("drisc_tiles") = FlatDriscTiles::both,
        R"doc(
        The flat expert's DRISC sender topology for ``n_slots`` gate/up weight slots: the senders' L1 working base,
        and per slot its DRAM tile (NoC xy, packed x << 16 | y), job index, alternate tile / job (switch mode; 0: none)
        and the tile that actually sends it (NOC0 grid coordinates, NoC). Introspection / benchmarks.
        )doc");

    mod.def(
        "flat_gu_placement_report",
        [](MeshDevice& mesh_device,
           uint32_t hidden,
           uint32_t intermediate,
           uint32_t experts_per_chip,
           uint32_t num_global_experts,
           uint32_t max_tokens_per_expert,
           bool weights_bf8,
           uint32_t pin,
           FlatLayoutOptions layout) {
            if (layout.gu_source != FlatGuSource::auto_hybrid) {
                layout.gu_source = FlatGuSource::auto_tensix;
            }
            const auto cfg = make_flat_routed_expert_config(
                hidden, intermediate, experts_per_chip, num_global_experts, max_tokens_per_expert, weights_bf8, pin,
                layout);
            const auto plan = flat_routed_expert_plan(mesh_device, cfg);
            const auto t = flat_gu_traffic(*plan);
            const bool hybrid = layout.gu_source == FlatGuSource::auto_hybrid;
            using clk = std::chrono::steady_clock;
            // every chip's geometry, deduplicated
            std::vector<tt::tt_metal::distributed::MeshCoordinate> coords;
            std::vector<FlatGuGeom> geoms;
            std::vector<uint32_t> topo_of;
            for (const auto id : mesh_device.get_device_ids()) {
                const auto coord = mesh_device.get_view().find_device(id);
                FlatGuGeom g = flat_gu_geom(mesh_device, *plan, coord);
                uint32_t k = 0;
                while (k < geoms.size() && geoms[k].key() != g.key()) {
                    ++k;
                }
                if (k == geoms.size()) {
                    geoms.push_back(std::move(g));
                }
                coords.push_back(coord);
                topo_of.push_back(k);
            }
            std::vector<FlatGuAssign> own;
            double ms_max = 0.0;
            for (const auto& g : geoms) {
                const auto t0 = clk::now();
                own.push_back(flat_gu_search({&g}, t, plan->rg, hybrid, kFlatGuIters));
                ms_max = std::max(ms_max, std::chrono::duration<double, std::milli>(clk::now() - t0).count());
            }
            std::vector<const FlatGuGeom*> gp;
            for (const auto& g : geoms) {
                gp.push_back(&g);
            }
            const auto t1 = clk::now();
            const FlatGuAssign comp = flat_gu_search(gp, t, plan->rg, hybrid, kFlatGuIters, kFlatGuCompromiseRestarts);
            const double ms_comp = std::chrono::duration<double, std::milli>(clk::now() - t1).count();
            const FlatGuAssign& ref = own[topo_of[0]];  // (the reference chip: the first local one)
            const double scale = experts_per_chip, us = 1e6 / (64.0 * 1.35e9);
            nb::list chips;
            for (size_t i = 0; i < coords.size(); ++i) {
                const auto& g = geoms[topo_of[i]];
                const auto& mine = own[topo_of[i]];
                uint32_t diff = 0;
                for (size_t c = 0; c < mine.bank.size(); ++c) {
                    diff += mine.bank[c] != ref.bank[c] || mine.how[c] != ref.how[c];
                }
                nb::dict d;
                d["coord"] = nb::make_tuple(coords[i][0], coords[i][1]);
                d["topology"] = topo_of[i];
                d["own_us"] = flat_gu_load(g, mine, t).max_link * scale * us;
                d["reference_us"] = flat_gu_load(g, ref, t).max_link * scale * us;
                d["compromise_us"] = flat_gu_load(g, comp, t).max_link * scale * us;
                // (what flat_routed_expert_chip_plan takes: the chip's own search or the compromise, the lighter)
                d["chosen_us"] = std::min(flat_gu_load(g, mine, t).max_link, flat_gu_load(g, comp, t).max_link) * scale * us;
                d["cells_differing_from_reference"] = diff;
                d["own_words"] = flat_gu_words(*plan, mine);
                chips.append(d);
            }
            nb::dict out;
            out["topologies"] = geoms.size();
            out["chips"] = chips;
            out["search_ms_max"] = ms_max;
            out["compromise_search_ms"] = ms_comp;
            out["compromise_words"] = flat_gu_words(*plan, comp);
            out["bank_floor_us"] = (2.0 * t.gu_core * flat_gu_cells(*plan).size() / 2.0 + t.down + t.x) * scale /
                                   plan->banks / 64e9 * 1e6;
            return out;
        },
        nb::arg("mesh_device"),
        nb::arg("hidden"),
        nb::arg("intermediate"),
        nb::arg("experts_per_chip"),
        nb::arg("num_global_experts"),
        nb::arg("max_tokens_per_expert"),
        nb::kw_only(),
        nb::arg("weights_bf8") = false,
        nb::arg("pin") = 1,
        nb::arg("layout") = FlatLayoutOptions{},
        R"doc(
        The flat expert's link-load gate/up placement per chip (host only, also on a mock cluster): the distinct chip
        geometries, and per chip the modelled busiest NoC link (us at 64 B / cycle, 1.35 GHz, weights + x + down
        reads) with its own placement, with the reference chip's, and with one compromise placement for all; the
        search times (ms).
        )doc");

    nb::class_<FlatDriscSenders>(mod, "FlatDriscSenders", R"doc(
        The flat routed expert's persistent DRISC gate/up weight senders on the mesh device's programmable DRAM cores
        (Blackhole). ``FlatDriscSenders.get(mesh_device, plan.drisc_n, plan.drisc_tiles)`` launches them on first use and shares them
        while held; they stop when the last holder drops them, on ``stop()``, or when the mesh device closes.
    )doc")
        .def_static(
            "supported",
            [](const MeshDevice& mesh_device) {
                return tt::tt_metal::experimental::AreDramCoreKernelsSupported(mesh_device);
            },
            nb::arg("mesh_device"),
            "Whether the mesh device can run them (programmable DRAM cores, the streaming profiler off).")
        .def_static(
            "get",
            [](MeshDevice& mesh_device, uint32_t n_slots, FlatDriscTiles drisc_tiles, FlatDriscProbe drisc_probe) {
                return FlatDriscSenders::get(mesh_device, n_slots, drisc_tiles, drisc_probe);
            },
            nb::arg("mesh_device"),
            nb::arg("n_slots"),
            nb::arg("drisc_tiles") = FlatDriscTiles::both,
            nb::arg("drisc_probe") = FlatDriscProbe::none,
            "The running senders for ``n_slots`` slots and ``drisc_tiles`` (plan.drisc_tiles); a set running for "
            "another tile choice is stopped first.")
        .def("stop", &FlatDriscSenders::stop, "Finishes the mesh's command queue, then stops the senders.")
        .def_prop_ro("running", &FlatDriscSenders::running)
        .def_prop_ro("n_slots", &FlatDriscSenders::n_slots)
        .def_prop_ro("drisc_tiles", &FlatDriscSenders::tiles)
        .def_prop_ro("drisc_probe", &FlatDriscSenders::probe)
        .def(
            "read_prof",
            [](const FlatDriscSenders& s) {
                std::vector<std::tuple<uint32_t, uint32_t, std::vector<uint32_t>>> out;
                for (auto& [c, w] : s.read_prof()) {
                    out.emplace_back(static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y), std::move(w));
                }
                return out;
            },
            "Probes: each sender's counters (bank, tile index, words; se_drisc_gu.cpp SE_DRISC_PROF_*).");

    ttnn::bind_function<"flat_routed_expert", "ttnn.experimental.moe_block.">(
        mod,
        R"doc(
        Flat spatially pipelined routed experts (Blackhole): every local expert's GLU FFN of a chip in ONE program
        that streams each expert's weights once through a spatial pipeline (the DRAM cores' DRISC senders feed the
        gate/up cores, x relays read / tilize the row-major dispatch buffer and feed it along unicast snakes, down
        cores write y at each expert's region). The routing's counts / regions rows are read on device, so one cached
        program serves every routing.

            y_e = act(x_e @ Wg_e, x_e @ Wu_e) @ Wd_e   for every local expert e

        Args:
            dispatched_buffer (ttnn.Tensor): [rows * P, H / P] bf16 ROW_MAJOR, DRAM interleaved (P: x_pages_per_row).
            expert_token_counts (ttnn.Tensor): [1, num_global_experts] uint32 ROW_MAJOR, DRAM interleaved.
            expert_region_offsets (ttnn.Tensor): [1, num_global_experts] uint32 ROW_MAJOR: each expert's first row.
            global_expert_ids (ttnn.Tensor): [experts_per_chip] uint32 ROW_MAJOR, DRAM (per device).
            gate_up_weights (ttnn.Tensor): the plan's gate/up bank layout (bfloat4_b or bfloat8_b).
            down_weights (ttnn.Tensor): the plan's down bank layout.
            done_words (ttnn.Tensor): create_flat_routed_expert_done_words(mesh_device, plan).
            drisc_senders (FlatDriscSenders): the running senders (FlatDriscSenders.get(mesh_device, plan.drisc_n)).
            intermediate (int): the expert intermediate size (per device).
            max_tokens_per_expert (int): the dispatch capacity per expert (>= 256).

        Keyword Args:
            activation (int): 0 SiLU, 1 SwiGLU-OAI, 2 SiTU, 3 clamped SiLU, 4 GELU-tanh.
            pin (int): pin the largest expert's weights in chunks of >= pin sub-blocks (0: off).
            token_index (ttnn.Tensor, optional): [1, rows] uint32 ROW_MAJOR DRAM; flat row r reads x row
                token_index[r] (x is then e.g. the all-gathered tokens); y has token_index's rows.
            x_pages_per_row (int): x stores each token row as this many consecutive pages.
            y_row_major (bool): y as row-major bf16 [rows, H], pack-untilized on the down cores.
            down_fp32 (bool): accumulate the down projection in fp32 DEST.
            pack_stochastic_rounding (bool): stochastic rounding in the compute kernels' packers.
            y_bf16 (bool): y as bf16 tiles (tiled y only).
            math_fidelity (int): the expert matmuls' fidelity (0 LoFi: exact for bfp4 weights; 2 HiFi2 for bfp8).
            layout (FlatLayoutOptions): the layout / program choices; must match the plan the weights were laid out
                with.
            output (ttnn.Tensor, optional): a preallocated y.

        Returns:
            ttnn.Tensor: y [rows, H] bfp8 TILE (bf16 TILE with y_bf16, bf16 ROW_MAJOR with y_row_major), written at
            the active experts' rows.
        )doc",
        &flat_routed_expert,
        nb::arg("dispatched_buffer").noconvert(),
        nb::arg("expert_token_counts").noconvert(),
        nb::arg("expert_region_offsets").noconvert(),
        nb::arg("global_expert_ids").noconvert(),
        nb::arg("gate_up_weights").noconvert(),
        nb::arg("down_weights").noconvert(),
        nb::arg("done_words").noconvert(),
        nb::arg("drisc_senders"),
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
        nb::arg("y_bf16") = false,
        nb::arg("math_fidelity") = 0,
        nb::arg("layout") = FlatLayoutOptions{},
        nb::arg("output") = nb::none());
}

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
