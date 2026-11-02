// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common_nanobind.hpp"

#include <limits>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn-nanobind/export_enum.hpp"

#include "l1_budget.hpp"
#include "mesh_routing.hpp"
#include "moe_block_config.hpp"
#include "moe_block_resources.hpp"
#include "port_placement.hpp"

namespace ttnn::operations::experimental::moe_block {

namespace nanobind_detail {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::distributed::MeshDevice;

// T(**kwargs): default-construct, then set each keyword as an attribute (unknown names raise).
template <typename T>
void def_kwargs_init(nb::class_<T>& cls) {
    cls.def(
        "__init__",
        [](T* self, const nb::kwargs& kwargs) {
            T value{};
            nb::object proxy = nb::cast(&value, nb::rv_policy::reference);
            for (auto [k, v] : kwargs) {
                if (!nb::hasattr(proxy, k)) {
                    throw nb::attribute_error((std::string(nb::type_name(proxy.type()).c_str()) + " has no field '" +
                                               nb::cast<std::string>(k) + "'")
                                                  .c_str());
                }
                nb::setattr(proxy, k, v);
            }
            new (self) T(value);
        },
        "Default values; any field can be given as a keyword.");
    cls.def(nb::self == nb::self);
}

Dir dir_of(const std::string& name) {
    for (Dir d : kDirs) {
        if (dir_name(d) == name) {
            return d;
        }
    }
    throw nb::value_error(("unknown direction '" + name + "' (X+, X-, Y+, Y-)").c_str());
}

nb::tuple chip_tuple(ChipCoord c) { return nb::make_tuple(c.r, c.c); }
ChipCoord chip_of(const std::pair<uint32_t, uint32_t>& p) { return {p.first, p.second}; }

// sparse_all_gather_2d.mesh_plan's chips: {(r, c): {"X+": {"peer": (r, c), "links": [...]}, ...}}
nb::dict plan_dict(const MeshPlan& plan) {
    nb::dict out;
    for (const auto& [ch, ports] : plan.chips) {
        nb::dict d;
        for (Dir dir : kDirs) {
            if (const auto& p = ports[static_cast<size_t>(dir)]) {
                nb::dict port;
                port["peer"] = chip_tuple(p->peer);
                port["links"] = nb::cast(p->links);
                d[nb::str(dir_name(dir).data(), dir_name(dir).size())] = port;
            }
        }
        out[chip_tuple(ch)] = d;
    }
    return out;
}

nb::dict luts_dict(const PortLuts& luts) {
    nb::dict lut, n;
    for (const auto& [ch, l] : luts.lut) {
        lut[chip_tuple(ch)] = nb::cast(std::vector<uint32_t>(l.begin(), l.end()));
    }
    for (const auto& [ch, k] : luts.num_ports) {
        n[chip_tuple(ch)] = k;
    }
    nb::dict out;
    out["lut"] = lut;
    out["num_ports"] = n;
    return out;
}

// {((r, c), "X+", link index): CoreCoord}
nb::dict ports_dict(const PortCores& ports) {
    nb::dict out;
    for (const auto& [k, core] : ports) {
        out[nb::make_tuple(chip_tuple(k.chip), nb::str(dir_name(k.dir).data()), k.link_index)] = nb::cast(core);
    }
    return out;
}

void bind_config(nb::module_& mod) {
    export_enum<PortPlacement>(mod, "PortPlacement");
    export_enum<ExpertActivationKind>(mod, "ExpertActivationKind");
    export_enum<RouterScoreFunc>(mod, "RouterScoreFunc");
    export_enum<ResidualLayout>(mod, "ResidualLayout");

    nb::class_<CombineTuning> tuning(mod, "CombineTuning", R"doc(
        The combine's tuning switches (the reference's SPC_* environment), today's defaults. All structural.

        snoc: 1 the sender on NoC1 (default), 0 on NoC0. la: look-ahead rows (0/1 strict order, max 64).
        far: with la > 1, rows going >= far hops first. credauto: credits only at the end when the pools hold the call.
        credthr: credit return threshold (8). credidle / credage: idle credit batching (0: the combine's choice).
        gbgate: the busy-pass credit scan gated on credthr. place: the port placement (PortPlacement.Noc1).
    )doc");
    def_kwargs_init(tuning);
    tuning.def_rw("snoc", &CombineTuning::snoc)
        .def_rw("la", &CombineTuning::la)
        .def_rw("far", &CombineTuning::far)
        .def_rw("credauto", &CombineTuning::credauto)
        .def_rw("credthr", &CombineTuning::credthr)
        .def_rw("credidle", &CombineTuning::credidle)
        .def_rw("credage", &CombineTuning::credage)
        .def_rw("gbgate", &CombineTuning::gbgate)
        .def_rw("place", &CombineTuning::place);

    nb::class_<ExpertActivation> act(mod, "ExpertActivation", "act(gate) * up of an expert.");
    def_kwargs_init(act);
    act.def_rw("kind", &ExpertActivation::kind)
        .def_rw("limit", &ExpertActivation::limit)
        .def_rw("alpha", &ExpertActivation::alpha)
        .def_rw("beta", &ExpertActivation::beta)
        .def_rw("linear_beta", &ExpertActivation::linear_beta)
        .def("validate", &ExpertActivation::validate);

    nb::class_<RouterSpec> router(mod, "RouterSpec", "The router (MoEModel.router).");
    def_kwargs_init(router);
    router.def_rw("score_func", &RouterSpec::score_func)
        .def_rw("route_scale", &RouterSpec::route_scale)
        .def_rw("normalize_topk", &RouterSpec::normalize_topk)
        .def_rw("expert_groups", &RouterSpec::expert_groups)
        .def_rw("topk_groups", &RouterSpec::topk_groups)
        .def_rw("fold", &RouterSpec::fold);

    nb::class_<MoEModelDims> dims(mod, "MoEModelDims", R"doc(
        The model's dims (MoEModel; K3Dims H/latent/I/E/K is the subset the exchange carries).
    )doc");
    def_kwargs_init(dims);
    dims.def_rw("num_experts", &MoEModelDims::num_experts)
        .def_rw("top_k", &MoEModelDims::top_k)
        .def_rw("emb_dim", &MoEModelDims::emb_dim)
        .def_rw("routed_expert_hidden_dim", &MoEModelDims::routed_expert_hidden_dim)
        .def_rw("routed_expert_activation", &MoEModelDims::routed_expert_activation)
        .def_rw("router", &MoEModelDims::router)
        .def_rw("shared_expert_hidden_dim", &MoEModelDims::shared_expert_hidden_dim)
        .def_rw("shared_expert_activation", &MoEModelDims::shared_expert_activation)
        .def_rw("latent_dim", &MoEModelDims::latent_dim)
        .def_rw("latent_norm", &MoEModelDims::latent_norm)
        .def_rw("norm_eps", &MoEModelDims::norm_eps)
        .def_prop_ro("routed_width", &MoEModelDims::routed_width)
        .def("validate", &MoEModelDims::validate);

    nb::class_<MoEDtypes> dtypes(mod, "MoEDtypes", "Weight / activation dtypes and fidelities.");
    def_kwargs_init(dtypes);
    dtypes.def_rw("routed_weights", &MoEDtypes::routed_weights)
        .def_rw("shared_weights", &MoEDtypes::shared_weights)
        .def_rw("shared_activations", &MoEDtypes::shared_activations)
        .def_rw("shared_fidelity", &MoEDtypes::shared_fidelity)
        .def_rw("projection_weights", &MoEDtypes::projection_weights)
        .def_rw("tail_fidelity", &MoEDtypes::tail_fidelity)
        .def_rw("router_weights", &MoEDtypes::router_weights);

    nb::class_<MoEBlockConfig> cfg(mod, "MoEBlockConfig", R"doc(
        The MoE block's options (MoEOptions) and the switches the reference read from the environment.
        Defaults are today's production defaults.
    )doc");
    def_kwargs_init(cfg);
    cfg.def_rw("experts_during_dispatch", &MoEBlockConfig::experts_during_dispatch)
        .def_rw("deterministic", &MoEBlockConfig::deterministic)
        .def_rw("combine_snap", &MoEBlockConfig::combine_snap)
        .def_rw("head_dram_fallback", &MoEBlockConfig::head_dram_fallback)
        .def_rw("router_split", &MoEBlockConfig::router_split)
        .def_rw("residual", &MoEBlockConfig::residual)
        .def_rw("num_links", &MoEBlockConfig::num_links)
        .def_rw("chunk_tokens", &MoEBlockConfig::chunk_tokens)
        .def_rw("zigzag", &MoEBlockConfig::zigzag)
        .def_rw("persist_l1_small", &MoEBlockConfig::persist_l1_small)
        .def_rw("words_slab_bytes", &MoEBlockConfig::words_slab_bytes)
        .def_rw("combine_tuning", &MoEBlockConfig::combine_tuning)
        .def_prop_ro("handoff_bytes", &MoEBlockConfig::handoff_bytes)
        .def_prop_ro("snap_bits", &MoEBlockConfig::snap_bits)
        .def("validate", &MoEBlockConfig::validate);

    mod.def("snap_bits", &snap_bits, nb::arg("q").none(), "fp32 bits of 1.5 x 2^23 q (0 for None): __SNAP__.");
}

void bind_routing(nb::module_& mod) {
    using Chip = std::pair<uint32_t, uint32_t>;
    auto topo = [](uint32_t R, uint32_t C, bool wrap_x, bool wrap_y) {
        return MeshTopology{.R = R, .C = C, .wrap_x = wrap_x, .wrap_y = wrap_y};
    };
    mod.def(
        "route",
        [topo](Chip src, Chip home, uint32_t R, uint32_t C, bool wrap_x, bool wrap_y) {
            nb::list out;
            for (const auto& h : route(chip_of(src), chip_of(home), topo(R, C, wrap_x, wrap_y))) {
                out.append(nb::make_tuple(chip_tuple(h.chip), nb::str(dir_name(h.dir).data())));
            }
            return out;
        },
        nb::arg("src"),
        nb::arg("home"),
        nb::arg("R"),
        nb::arg("C"),
        nb::arg("wrap_x"),
        nb::arg("wrap_y"),
        "[((r, c), dir)] hops from src to home: Y first, then X; a ring tie by the source's parity, as main's fabric "
        "(sparse_combine_l1.route, tie_mode 2).");
    mod.def(
        "first_hop",
        [topo](Chip src, Chip dst, uint32_t R, uint32_t C, bool wrap_x, bool wrap_y) -> std::optional<std::string> {
            const auto d = first_hop(chip_of(src), chip_of(dst), topo(R, C, wrap_x, wrap_y));
            return d ? std::optional<std::string>(dir_name(*d)) : std::nullopt;
        },
        nb::arg("src"),
        nb::arg("dst"),
        nb::arg("R"),
        nb::arg("C"),
        nb::arg("wrap_x"),
        nb::arg("wrap_y"),
        "The first hop's direction ('X+', 'X-', 'Y+', 'Y-'), None for src == dst.");
    mod.def(
        "first_hop_idx",
        [topo](uint32_t src, uint32_t dst, uint32_t R, uint32_t C, bool wrap_x, bool wrap_y) {
            return first_hop_idx(src, dst, topo(R, C, wrap_x, wrap_y));
        },
        nb::arg("src"),
        nb::arg("dst"),
        nb::arg("R"),
        nb::arg("C"),
        nb::arg("wrap_x"),
        nb::arg("wrap_y"),
        "DIRS index (0 X+, 1 X-, 2 Y+, 3 Y-) of the first hop between linear chip indices.");
    mod.def("link_of", &link_of, nb::arg("acc"), nb::arg("src"), nb::arg("home"), nb::arg("L"));
    mod.def("hbit", &hbit, nb::arg("t"), nb::arg("k"));
    mod.def("route_words", &route_words, nb::arg("N"));
    mod.def("span_entry", &span_entry, nb::arg("n"), nb::arg("wrap"), nb::arg("pos"), nb::arg("tmask"), nb::arg("tie"));
    mod.def(
        "span_table",
        [](uint32_t n, bool wrap) {
            const auto t = span_table(n, wrap);
            return std::vector<uint32_t>(t.begin(), t.end());
        },
        nb::arg("n"),
        nb::arg("wrap"));
    mod.def(
        "ring_table",
        [](uint32_t n, bool wrap) {
            const auto t = ring_table(n, wrap);
            return std::vector<uint32_t>(t.begin(), t.end());
        },
        nb::arg("n"),
        nb::arg("wrap"));
    mod.def(
        "span_subs",
        [](uint32_t R, uint32_t C, bool wrap_x, bool wrap_y) {
            const auto s = span_subs(R, C, wrap_x, wrap_y);
            nb::dict out;
            out["wrap_x"] = s.wrap_x;
            out["wrap_y"] = s.wrap_y;
            out["xring"] = nb::cast(std::vector<uint32_t>(s.xring.begin(), s.xring.end()));
            out["yring"] = nb::cast(std::vector<uint32_t>(s.yring.begin(), s.yring.end()));
            return out;
        },
        nb::arg("R"),
        nb::arg("C"),
        nb::arg("wrap_x"),
        nb::arg("wrap_y"),
        "The dispatch kernels' span constants: wrap flags and the X / Y ring tables.");
    mod.def("axis_reach", &axis_reach, nb::arg("n"), nb::arg("wrap"));
    mod.def(
        "manifest_layout",
        [](uint32_t R, uint32_t C, bool wrap_x, bool wrap_y, uint32_t S) {
            const auto m = manifest_layout(R, C, wrap_x, wrap_y, S);
            return std::make_tuple(m.ring1_offset, m.ring2_offset, m.total_entries);
        },
        nb::arg("R"),
        nb::arg("C"),
        nb::arg("wrap_x"),
        nb::arg("wrap_y"),
        nb::arg("S"),
        "(ring 1's offset, ring 2's offset, total) in 16-byte manifest entries.");
    mod.def(
        "mesh_plan_uniform",
        [](uint32_t R, uint32_t C, bool torus_x, bool torus_y, uint32_t num_links) {
            return plan_dict(mesh_plan_uniform(MeshTopology::of(R, C, torus_x, torus_y), num_links));
        },
        nb::arg("R"),
        nb::arg("C"),
        nb::arg("torus_x"),
        nb::arg("torus_y"),
        nb::arg("num_links"),
        "Host-only mesh plan: every neighbour on links 0..num_links-1 ({(r, c): {dir: {peer, links}}}).");
    mod.def(
        "port_luts_uniform",
        [](uint32_t R, uint32_t C, bool torus_x, bool torus_y, uint32_t num_links) {
            return luts_dict(port_luts(mesh_plan_uniform(MeshTopology::of(R, C, torus_x, torus_y), num_links)));
        },
        nb::arg("R"),
        nb::arg("C"),
        nb::arg("torus_x"),
        nb::arg("torus_y"),
        nb::arg("num_links"),
        "port_luts of mesh_plan_uniform: {'lut': {(r, c): [8]}, 'num_ports': {(r, c): n}}.");
    mod.def(
        "mesh_plan",
        [](const MeshDevice& mesh, uint32_t num_links, bool torus_x, bool torus_y) {
            return plan_dict(mesh_plan(mesh, num_links, torus_x, torus_y));
        },
        nb::arg("mesh_device"),
        nb::arg("num_links"),
        nb::arg("torus_x"),
        nb::arg("torus_y"),
        "The mesh plan from the fabric (sparse_all_gather_2d.mesh_plan).");
    mod.def(
        "place_ports",
        [](MeshDevice& mesh,
           uint32_t num_links,
           bool torus_x,
           bool torus_y,
           PortPlacement mode,
           const std::optional<std::vector<CoreCoord>>& allowed) {
            return ports_dict(place_ports(mesh, mesh_plan(mesh, num_links, torus_x, torus_y), mode, allowed));
        },
        nb::arg("mesh_device"),
        nb::arg("num_links"),
        nb::arg("torus_x"),
        nb::arg("torus_y"),
        nb::arg("mode") = PortPlacement::Noc1,
        nb::arg("allowed") = nb::none(),
        "{((r, c), dir, link index): CoreCoord}: a port core per direction and link next to its Ethernet core.");
}

void bind_resources(nb::module_& mod) {
    nb::class_<MoEBlockResources>(mod, "MoEBlockResources", R"doc(
        Everything the MoE block's ops share across calls: persistent L1 / DRAM words, global semaphores, port
        placement, plans. Create once per (mesh, model, config) after the mesh is open and the fabric set, before any
        per-call allocation; release() (or drop it) before closing the mesh. Ops take it as an argument and
        get-or-create their "<op>/<name>" slots; freeze() before trace capture.
    )doc")
        .def(
            nb::new_(
                [](MeshDevice& mesh_device,
                   const MoEModelDims& dims,
                   const MoEBlockConfig& config,
                   const MoEDtypes& dtypes) { return MoEBlockResources::create(mesh_device, dims, config, dtypes); }),
            nb::arg("mesh_device"),
            nb::arg("dims"),
            nb::arg("config") = MoEBlockConfig{},
            nb::arg("dtypes") = MoEDtypes{})
        .def_prop_ro("config", [](const MoEBlockResources& r) { return r.config(); })
        .def_prop_ro("dtypes", [](const MoEBlockResources& r) { return r.dtypes(); })
        .def_prop_ro("dims", [](const MoEBlockResources& r) { return r.dims(); })
        .def_prop_ro("plan_key", &MoEBlockResources::plan_key)
        .def_prop_ro("persistent_buffer_type", &MoEBlockResources::persistent_buffer_type)
        .def_prop_ro(
            "topology",
            [](const MoEBlockResources& r) {
                const auto& t = r.topology();
                nb::dict out;
                out["R"] = t.R;
                out["C"] = t.C;
                out["wrap_x"] = t.wrap_x;
                out["wrap_y"] = t.wrap_y;
                return out;
            })
        .def_prop_ro("mesh_plan", [](const MoEBlockResources& r) { return plan_dict(r.mesh_plan()); })
        .def_prop_ro("port_luts", [](const MoEBlockResources& r) { return luts_dict(r.port_luts()); })
        .def(
            "ports",
            [](MoEBlockResources& r,
               std::optional<PortPlacement> mode,
               const std::optional<std::vector<CoreCoord>>& allowed,
               const std::optional<std::vector<std::string>>& dirs) {
                if (!dirs) {
                    return ports_dict(r.ports(mode, allowed));
                }
                std::vector<Dir> ds;
                for (const auto& name : *dirs) {
                    ds.push_back(dir_of(name));
                }
                return ports_dict(r.ports(ds, mode, allowed));
            },
            nb::arg("mode") = nb::none(),
            nb::arg("allowed") = nb::none(),
            nb::arg("dirs") = nb::none(),
            "{((r, c), dir, link index): CoreCoord}, memoized. dirs: place only these directions' ports, e.g. "
            "[\"X+\", \"X-\"] (default: all).")
        .def(
            "persistent_tensor",
            [](MoEBlockResources& r, const std::string& name, const tt::tt_metal::TensorSpec& spec) {
                return r.persistent_tensor(name, spec);
            },
            nb::arg("name"),
            nb::arg("spec"))
        .def(
            "persistent_words",
            [](MoEBlockResources& r, const std::string& name, const CoreRangeSet& cores, uint32_t words) {
                return r.persistent_words(name, cores, words);
            },
            nb::arg("name"),
            nb::arg("cores"),
            nb::arg("words"))
        .def(
            "global_semaphore",
            [](MoEBlockResources& r, const std::string& name, const CoreRangeSet& cores, uint32_t initial_value) {
                return r.global_semaphore(name, cores, initial_value);
            },
            nb::arg("name"),
            nb::arg("cores"),
            nb::arg("initial_value") = 0)
        .def("semaphore_address", &MoEBlockResources::semaphore_address, nb::arg("name"))
        .def("has", &MoEBlockResources::has, nb::arg("name"))
        .def("names", &MoEBlockResources::names)
        .def(
            "words_layout",
            [](const MoEBlockResources& r) {
                nb::list out;
                for (const auto& w : r.words_layout()) {
                    nb::list cores;
                    for (const auto& c : corerange_to_cores(w.cores)) {
                        cores.append(nb::make_tuple(c.x, c.y));
                    }
                    nb::dict d;
                    d["name"] = w.name;
                    d["cores"] = cores;
                    d["bytes"] = w.bytes;
                    d["address"] = w.address;
                    d["packed"] = w.packed;
                    out.append(d);
                }
                return out;
            },
            "[{name, cores [(x, y)], bytes (a core), address, packed}] a persistent_words name.")
        .def(
            "words_slab",
            [](const MoEBlockResources& r) -> nb::object {
                const auto s = r.words_slab();
                if (!s) {
                    return nb::none();
                }
                nb::dict d;
                d["address"] = s->address;
                d["capacity"] = s->capacity;
                d["used"] = s->used;
                return d;
            },
            "{address, capacity, used} of the packed persistent words' L1_SMALL slab (None before its first slot).")
        .def("freeze", &MoEBlockResources::freeze)
        .def_prop_ro("frozen", &MoEBlockResources::frozen)
        .def("release", &MoEBlockResources::release);

    mod.def(
        "free_cb_l1_bytes",
        [](const MeshDevice& mesh_device, uint32_t margin_bytes, uint32_t cap_bytes) {
            return free_cb_l1_bytes(mesh_device, margin_bytes, cap_bytes);
        },
        nb::arg("mesh_device"),
        nb::arg("margin_bytes") = kL1KeptLater,
        nb::arg("cap_bytes") = std::numeric_limits<uint32_t>::max(),
        "L1 bytes left for program CBs on every compute core: the lowest occupied L1 address less the L1 base, less "
        "margin_bytes, rounded down to 64 B, at most cap_bytes.");
}

}  // namespace nanobind_detail

void bind_common(nb::module_& mod) {
    nanobind_detail::bind_config(mod);
    nanobind_detail::bind_routing(mod);
    nanobind_detail::bind_resources(mod);
}

}  // namespace ttnn::operations::experimental::moe_block
