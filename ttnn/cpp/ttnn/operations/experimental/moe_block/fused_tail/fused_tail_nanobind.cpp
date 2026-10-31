// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_tail_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"

#include "fused_tail.hpp"
#include "device/fused_tail_plan.hpp"

namespace ttnn::operations::experimental::moe_block::fused_tail {

namespace fused_tail_nanobind_detail {

nb::list units(const std::vector<Unit>& us) {
    nb::list out;
    for (const auto& u : us) {
        out.append(nb::make_tuple(u.pos, u.part));
    }
    return out;
}

const char* dir_str(TailDir d) { return d == TailDir::Fwd ? "fwd" : "bwd"; }
TailDir dir_of(const std::string& s) {
    TT_FATAL(s == "fwd" || s == "bwd", "fused_tail: a direction is 'fwd' or 'bwd', got '{}'", s);
    return s == "fwd" ? TailDir::Fwd : TailDir::Bwd;
}

// The host plans as plain Python values, for the parity test against the reference (no device).
void bind_host_math(nb::module_& mod) {
    mod.def(
        "fused_tail_gather_schedule",
        [](uint32_t p, uint32_t G, bool ring, bool balance) {
            const auto [fwd, bwd] = gather_schedule(p, G, ring, balance);
            return nb::make_tuple(units(fwd), units(bwd));
        },
        nb::arg("p"),
        nb::arg("G"),
        nb::arg("ring"),
        nb::arg("balance"),
        "fabric_all_gather_py._schedule: ([(pos, part)] toward p + 1, [(pos, part)] toward p - 1).");
    mod.def(
        "fused_tail_chunk_order",
        &chunk_order,
        nb::arg("p"),
        nb::arg("G"),
        nb::arg("ring"),
        "tail_engine.chunk_order.");
    mod.def(
        "fused_tail_a2a_schedule",
        [](uint32_t G,
           bool ring,
           uint32_t L,
           uint32_t NTB,
           uint32_t RT,
           uint32_t NBK,
           uint32_t RP,
           const nb::dict& ties) {
            std::map<std::pair<uint32_t, uint32_t>, TailDir> t;
            for (auto [k, v] : ties) {
                t[nb::cast<std::pair<uint32_t, uint32_t>>(k)] = dir_of(nb::cast<std::string>(v));
            }
            const TailSchedule sc(G, ring, L, NTB, RT, NBK, RP, std::move(t));
            nb::dict expect, readies, dests, runs;
            for (uint32_t p = 0; p < G; ++p) {
                for (TailDir d : {TailDir::Fwd, TailDir::Bwd}) {
                    dests[nb::make_tuple(p, dir_str(d))] = units(sc.dests(p, d));
                    for (uint32_t l = 0; l < L; ++l) {
                        const TailSchedule::Port port{p, d, l};
                        const auto key = nb::make_tuple(p, dir_str(d), l);
                        if (const uint32_t e = sc.expect(port)) {
                            expect[key] = e;
                        }
                        nb::list rs;
                        for (const auto& [xp, xd, xl] : sc.readies(port)) {
                            rs.append(nb::make_tuple(xp, dir_str(xd), xl));
                        }
                        if (nb::len(rs)) {
                            readies[key] = rs;
                        }
                    }
                }
            }
            for (uint32_t part = 0; part < 3; ++part) {
                for (uint32_t l = 0; l < L; ++l) {
                    runs[nb::make_tuple(part, l)] = sc.run_list(part, l);
                }
            }
            nb::dict out;
            out["expect"] = expect;
            out["readies"] = readies;
            out["dests"] = dests;
            out["run_list"] = runs;
            return out;
        },
        nb::arg("G"),
        nb::arg("ring"),
        nb::arg("L"),
        nb::arg("NTB"),
        nb::arg("RT"),
        nb::arg("NBK"),
        nb::arg("RP"),
        nb::arg("ties") = nb::dict(),
        "tail_a2a.TailSchedule as {expect, readies (sorted lists), dests, run_list}.");
    mod.def(
        "fused_tail_engine_geometry",
        [](uint32_t rows,
           uint32_t Hr,
           uint32_t Ho,
           uint32_t chunks,
           uint32_t GX,
           uint32_t GY,
           uint32_t wtb,
           uint32_t cap) {
            const auto g = TailEngineGeometry::make(rows, Hr, Ho, chunks, GX, GY, wtb, cap);
            nb::dict out;
            out["KT"] = g.KT;
            out["NTO"] = g.NTO;
            out["RT"] = g.RT;
            out["PAIRS"] = g.PAIRS;
            out["nst"] = g.nst;
            out["nown"] = g.nown;
            out["NSTMAX"] = g.NSTMAX;
            out["NOWNMAX"] = g.NOWNMAX;
            out["kh"] = g.kh;
            out["KH"] = g.KH;
            out["pn"] = g.pn;
            out["n0"] = g.n0;
            out["PNMAX"] = g.PNMAX;
            nb::list fin;
            for (const auto& f : g.fin) {
                fin.append(
                    nb::make_tuple(nb::make_tuple(f[0].first, f[0].second), nb::make_tuple(f[1].first, f[1].second)));
            }
            out["fin"] = fin;
            out["FNS"] = g.FNS;
            out["FNR"] = g.FNR;
            out["wk"] = g.wk;
            out["ak"] = g.ak;
            out["SHMAX"] = g.SHMAX;
            out["NSLOT"] = g.NSLOT;
            out["arena_bytes"] = g.arena_bytes;
            out["layout"] = g.layout(g.NSLOT).first;
            out["region_tiles"] = g.region_tiles();
            return out;
        },
        nb::arg("rows"),
        nb::arg("Hr"),
        nb::arg("Ho"),
        nb::arg("chunks"),
        nb::arg("GX"),
        nb::arg("GY"),
        nb::arg("wtb"),
        nb::arg("cap"),
        "TailEngine's geometry (TailEngine.__init__ / _layout) as a dict.");
    mod.def(
        "fused_tail_engine_relays",
        [](uint32_t p, uint32_t G, bool ring, uint32_t rows, uint32_t Hr, uint32_t Ho, uint32_t GX, uint32_t GY) {
            const auto g = TailEngineGeometry::make(rows, Hr, Ho, G, GX, GY, 1088, kArenaMax);
            return engine_relays(p, G, ring, g);
        },
        nb::arg("p"),
        nb::arg("G"),
        nb::arg("ring"),
        nb::arg("rows"),
        nb::arg("Hr"),
        nb::arg("Ho"),
        nb::arg("GX"),
        nb::arg("GY"),
        "TailEngine._relays of the chip at TP position p: {(cx, cy): [s | c0 << 8 | c1 << 16]}.");
}

}  // namespace fused_tail_nanobind_detail
using namespace fused_tail_nanobind_detail;

void bind_fused_tail(nb::module_& mod) {
    ttnn::bind_function<"fused_tail", "ttnn.experimental.moe_block.">(
        mod,
        R"doc(
        The MoE block's tail as one program a chip: the combine's owner rows (+ the shared expert's output) to the
        residual's layout [1, 1, S/SP, H/TP] (a mesh row's tokens, this chip's width slice). The TP group is the mesh
        row; a ring when the fabric wraps it.

        A latent model (resources.dims.latent_dim set): the latent all-gather over TP into the engine's L1, the own
        rows tilized there, out = rmsnorm(z) @ W_up[:, chip's columns] + shared_expert (the norm folded: its weight in
        w_up, 1 / rms applied to the fp32 sums). Otherwise: out = the TP all-to-all of tilize(y_own) + shared_expert.

        Args:
            y_own: the combine's owner rows [1, 1, S/N, Hr], bfloat16 ROW_MAJOR, DRAM interleaved.
            shared_expert: bfloat16 TILE, DRAM interleaved: [1, 1, S/SP, H/TP] with a latent (required),
                [1, 1, S/N, H] without (or None).

        Keyword args:
            resources: the block's MoEBlockResources (persistent words, plans; created once a mesh).
            w_up: a latent model's prepare_fused_tail_weights (required with a latent, None without).
            math_fidelity: the up projection's fidelity (HiFi2; never LoFi). The squares' sums run at HiFi4.
            output: a preallocated output of the result's spec.

        Returns:
            [1, 1, S/SP, H/TP] bfloat16 TILE, DRAM interleaved.

        The program is cached by structure (resources.plan_key, shapes, fidelity): a repeated call patches only the
        tensors' addresses. Persistent counters only count up or are re-armed by their waiters, so traced replays
        need no reset.
        )doc",
        &fused_tail,
        nb::arg("y_own"),
        nb::arg("shared_expert"),
        nb::kw_only(),
        nb::arg("resources"),
        nb::arg("w_up") = nb::none(),
        nb::arg("math_fidelity") = tt::tt_metal::MathFidelity::HiFi2,
        nb::arg("output") = nb::none());

    ttnn::bind_function<"prepare_fused_tail_weights", "ttnn.experimental.moe_block.">(
        mod,
        R"doc(
        A latent model's up projection for fused_tail, on device: latent_up [H, Hr] (host, float32 or bfloat16)
        split by output columns over TP (chip (r, c): columns [c H/TP, (c + 1) H/TP)), the rmsnorm weight latent_norm
        [Hr] folded in (W' = diag(g) W_up^T), laid out as the engine reads it (K-range regions of each engine column,
        one DRAM burst each). Upload once a layer and keep it.

        Args:
            latent_up: [H, Hr] host tensor.
            latent_norm: [Hr] host tensor.

        Keyword args:
            resources: the block's MoEBlockResources.
            dtype: bfloat8_b (default) or bfloat16.
        )doc",
        &prepare_fused_tail_weights,
        nb::arg("latent_up"),
        nb::arg("latent_norm"),
        nb::kw_only(),
        nb::arg("resources"),
        nb::arg("dtype") = tt::tt_metal::DataType::BFLOAT8_B);

    bind_host_math(mod);
}

}  // namespace ttnn::operations::experimental::moe_block::fused_tail
