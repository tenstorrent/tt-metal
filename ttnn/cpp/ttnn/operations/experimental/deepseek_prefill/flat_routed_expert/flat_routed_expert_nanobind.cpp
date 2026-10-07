// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/mesh_device.hpp>

#include "flat_routed_expert.hpp"
#include "flat_routed_expert_weights.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::detail {

namespace nb = nanobind;
namespace fre = ttnn::operations::experimental::deepseek_prefill::flat_routed_expert;

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
    uint32_t pin) {
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
            .pin = pin});
    nb::dict d;
    d["nsg"] = p->nsg;
    d["n_rd"] = p->readers.size();
    d["n_rd_sg"] = p->n_rd_sg;
    d["np"] = p->np;
    d["g"] = p->g;
    d["mt"] = p->mt;
    d["rg"] = p->rg;
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
    d["x_slots"] = p->x_slots;
    d["hbuf"] = p->hbuf;
    d["gu_rp"] = p->gu_rp;
    d["gu_l1acc"] = p->gu_l1acc;
    d["arena_tiles"] = p->arena_tiles;
    d["coords"] = cores(p->coords);
    d["readers"] = cores(p->readers);
    d["gu"] = cores(p->gu);
    d["relays"] = cores(p->relays);
    d["down"] = cores(p->down);
    return d;
}
}  // namespace

void bind_flat_routed_expert(nb::module_& mod) {
    ttnn::bind_function<"flat_routed_expert", "ttnn.experimental.deepseek_prefill.">(
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
        nb::arg("y_row_major") = false);

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
        "The flat_routed_expert layout plan (weight bank layout, coordinator cores for the done words).");

    mod.def(
        "flat_routed_expert_gather_tiles",
        [](const std::vector<ttnn::Tensor>& sources,
           const nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>& tile_map,
           const ttnn::Shape& shape,
           const tt::tt_metal::MemoryConfig& memory_config) {
            std::vector<int64_t> map(tile_map.data(), tile_map.data() + tile_map.size());
            nb::gil_scoped_release release;
            return fre::gather_host_tiles(sources, map, shape, memory_config);
        },
        nb::arg("sources"),
        nb::arg("tile_map"),
        nb::arg("shape"),
        nb::arg("memory_config"),
        R"doc(
        Host tensor built by copying whole packed tiles out of host source tensors (no unpacking: the sources'
        quantization is reused exactly). ``tile_map`` (1-D int64 numpy array) has one entry per destination tile, in row-major tile order of
        a destination shard: ``(source_index << 32) | source_tile`` or -1 for a zero tile; the same map is applied
        to every mesh shard. Sources: host, TILE, interleaved, one dtype, same mesh shards. Returns a host tensor of
        ``shape`` / ``memory_config`` per shard, sharded on dim 0 over the sources' mesh.
        )doc");
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
