// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common_nanobind.hpp"

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

#include "mesh_routing.hpp"
#include "moe_block_config.hpp"
#include "moe_block_resources.hpp"

namespace ttnn::operations::experimental::moe_block {

namespace nanobind_detail {

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

nb::tuple chip_tuple(ChipCoord c) { return nb::make_tuple(c.r, c.c); }

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

void bind_config(nb::module_& mod) {
    export_enum<ExpertActivationKind>(mod, "ExpertActivationKind");
    export_enum<RouterScoreFunc>(mod, "RouterScoreFunc");

    nb::class_<ExpertActivation> act(mod, "ExpertActivation", "act(gate) * up of an expert.");
    def_kwargs_init(act);
    act.def_rw("kind", &ExpertActivation::kind)
        .def_rw("limit", &ExpertActivation::limit)
        .def_rw("alpha", &ExpertActivation::alpha)
        .def_rw("beta", &ExpertActivation::beta)
        .def_rw("linear_beta", &ExpertActivation::linear_beta)
        .def("check", &ExpertActivation::check);

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
        .def("check", &MoEModelDims::check);

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
        The MoE block's options. Defaults are today's production defaults.
    )doc");
    def_kwargs_init(cfg);
    cfg.def_rw("deterministic", &MoEBlockConfig::deterministic)
        .def_rw("combine_snap", &MoEBlockConfig::combine_snap)
        .def_rw("head_dram_fallback", &MoEBlockConfig::head_dram_fallback)
        .def_rw("num_links", &MoEBlockConfig::num_links)
        .def_rw("chunk_tokens", &MoEBlockConfig::chunk_tokens)
        .def_rw("persist_l1_small", &MoEBlockConfig::persist_l1_small)
        .def_rw("words_slab_bytes", &MoEBlockConfig::words_slab_bytes)
        .def("check", &MoEBlockConfig::check);
}

void bind_routing(nb::module_& mod) {
    mod.def("route_words", &route_words, nb::arg("N"));
    mod.def("span_entry", &span_entry, nb::arg("n"), nb::arg("wrap"), nb::arg("pos"), nb::arg("tmask"), nb::arg("tie"));
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
}

void bind_resources(nb::module_& mod) {
    nb::class_<MoEBlockResources>(mod, "MoEBlockResources", R"doc(
        Everything the MoE block's ops share across calls: persistent L1 / DRAM words and tensors, port placement,
        plans. Create once per (mesh, model, config) after the mesh is open and the fabric set, before any
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
        .def("freeze", &MoEBlockResources::freeze)
        .def_prop_ro("frozen", &MoEBlockResources::frozen)
        .def("release", &MoEBlockResources::release);
}

}  // namespace nanobind_detail

void bind_common(nb::module_& mod) {
    nanobind_detail::bind_config(mod);
    nanobind_detail::bind_routing(mod);
    nanobind_detail::bind_resources(mod);
}

}  // namespace ttnn::operations::experimental::moe_block
