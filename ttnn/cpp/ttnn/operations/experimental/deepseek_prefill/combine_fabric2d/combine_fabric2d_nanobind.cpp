// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "combine_fabric2d.hpp"
#include "device/combine_fabric2d_program_factory.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::detail {
void bind_experimental_combine_fabric2d_operation(nb::module_& mod) {
    // One requested data movement. Bound as a named type rather than a bare tuple because it IS the op's
    // contract: the caller states what moves where, and everything about how the op achieves it (core
    // count, cable choice, which producer serves which movement) stays on the op's side of this line.
    nb::class_<CombineFabric2dMovement>(mod, "CombineFabric2dMovement")
        .def(
            "__init__",
            [](CombineFabric2dMovement* self,
               const std::vector<uint32_t>& src,
               uint32_t in_base_token,
               const std::vector<uint32_t>& dst,
               uint32_t out_base_token) {
                new (self) CombineFabric2dMovement{src, in_base_token, dst, out_base_token};
            },
            nb::arg("src"),
            nb::arg("in_base_token"),
            nb::arg("dst"),
            nb::arg("out_base_token"),
            R"doc(
            Copy `tokens_per_movement` tokens starting at `in_base_token` of device `src`'s input region
            to `tokens_per_movement` tokens starting at `out_base_token` of device `dst`'s output region,
            1:1 and in order. `src` and `dst` are mesh coordinates; the bases are in TOKENS (= pages), not
            bytes. `dst` must be a chip `src` has a fabric cable to.
            )doc")
        .def_rw("src", &CombineFabric2dMovement::src)
        .def_rw("in_base_token", &CombineFabric2dMovement::in_base_token)
        .def_rw("dst", &CombineFabric2dMovement::dst)
        .def_rw("out_base_token", &CombineFabric2dMovement::out_base_token)
        .def("__repr__", [](const CombineFabric2dMovement& m) {
            std::string s = "CombineFabric2dMovement(src=(";
            for (size_t i = 0; i < m.src.size(); i++) {
                s += (i ? "," : "") + std::to_string(m.src[i]);
            }
            s += "), in_base_token=" + std::to_string(m.in_base_token) + ", dst=(";
            for (size_t i = 0; i < m.dst.size(); i++) {
                s += (i ? "," : "") + std::to_string(m.dst[i]);
            }
            return s + "), out_base_token=" + std::to_string(m.out_base_token) + ")";
        });

    ttnn::bind_function<"combine_fabric2d", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Isolated FABRIC_2D transfer experiment op: chip-local DRAM -> eth -> neighbour chip's DRAM.

        The caller supplies two region tensors and a list of `CombineFabric2dMovement` descriptors
        saying what moves where. The op places one producer kernel per fabric eth core (`num_links`
        toward each neighbour along mesh `axis`, both directions, so 2 * num_links per device) plus a
        reader kernel beside it on the same core, and gives each pair one of that device's movements
        whose `dst` its cable reaches. The reader streams that movement's tokens out of local DRAM into
        an L1 ring over NOC_0 while the producer drains the ring to the peer chip's DRAM over NOC_1, one
        fabric packet per token. The token count is the caller's and is not bounded by L1. There is no
        receiver kernel and no application-level credit loop; the EDM sender-slot backpressure is the
        only throttle across the fabric.

        `input` and `output` are caller-owned interleaved uint32 ROW_MAJOR DRAM mesh tensors with one
        row per token. Every device needs exactly one movement per fabric cable it has; the op rejects
        a list that does not cover them, that names an unreachable `dst`, or whose output ranges
        overlap on a destination. Which of the equally-valid producers serves a given movement, and
        `num_l1_slots` (the ring depth), are internal details: results do not depend on either.
        Returns `output`.
        )doc",
        &combine_fabric2d,
        nb::arg("device"),
        nb::arg("input"),
        nb::arg("output"),
        nb::arg("movements"),
        nb::arg("num_links") = 2,
        nb::arg("tokens_per_movement") = 100,
        nb::arg("token_size_bytes") = 14336,
        nb::arg("axis") = 0,
        nb::arg("num_l1_slots") = 8,
        nb::arg("fwd_bump_every") = 8,
        nb::arg("stall_telemetry") = 0,
        nb::arg("topology") = nb::none());

    // Telemetry readback. Returns {"clock_mhz": int, "workers": [ {...}, ... ]} — plain Python data so
    // callers can format it however they like. The caller does NOT need to know where the worker cores
    // are: placement is recomputed here from (axis, num_links), which must match the op's run.
    mod.def(
        "combine_fabric2d_read_telemetry",
        [](ttnn::MeshDevice* mesh_device, uint32_t num_links, uint32_t axis) {
            const auto telem = read_telemetry(mesh_device, num_links, axis);
            nb::list workers;
            for (const auto& w : telem.workers) {
                nb::dict d;
                d["device_id"] = w.device_id;
                d["mesh_coord"] = w.mesh_coord;
                d["worker_logical"] = nb::make_tuple(w.worker_logical.x, w.worker_logical.y);
                d["worker_physical"] = nb::make_tuple(w.worker_physical.x, w.worker_physical.y);
                d["eth_logical"] = nb::make_tuple(w.eth_logical.x, w.eth_logical.y);
                d["eth_phys_x"] = w.eth_phys_x;
                d["link_idx"] = w.link_idx;
                d["relocated"] = w.relocated;
                d["peer_mesh_id"] = w.peer_mesh_id;
                d["peer_chip_id"] = w.peer_chip_id;
                d["peer_coord"] = w.peer_coord;
                d["valid"] = w.valid;
                d["tokens_sent"] = w.tokens_sent;
                d["token_size_bytes"] = w.token_size_bytes;
                d["num_l1_slots"] = w.num_l1_slots;
                d["batch"] = w.batch;
                d["t_start"] = w.t_start;
                d["t_first_send"] = w.t_first_send;
                d["t_last_send"] = w.t_last_send;
                d["t_drained"] = w.t_drained;
                d["t_kernel_start"] = w.t_kernel_start;
                d["t_kernel_end"] = w.t_kernel_end;
                d["edm_slots"] = w.edm_slots;
                d["drain_packets"] = w.drain_packets;
                d["out_base_page"] = w.out_base_page;
                d["wait_slot_cycles"] = w.wait_slot_cycles;
                d["issue_cycles"] = w.issue_cycles;
                d["ring_wait_cycles"] = w.ring_wait_cycles;
                workers.append(d);
            }
            nb::dict out;
            out["clock_mhz"] = telem.clock_mhz;
            out["workers"] = workers;
            return out;
        },
        nb::arg("device"),
        nb::arg("num_links") = 2,
        nb::arg("axis") = 0,
        R"doc(
        Read the per-worker CombineFabric2D telemetry out of L1 after the op has run. Recovers
        bandwidth without re-running under the profiler. `num_links` and `axis` must match the run.
        )doc");
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::detail
