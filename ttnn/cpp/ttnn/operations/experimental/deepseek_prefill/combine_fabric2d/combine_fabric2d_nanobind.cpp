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
    ttnn::bind_function<"combine_fabric2d", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Isolated FABRIC_2D transfer experiment op. For each fabric eth core (`num_links` toward
        each neighbor along mesh `axis`), one worker core in that eth core's physical column runs a
        producer on the writer RISC and a receiver on the reader RISC. Every link is full duplex:
        the producer sends `num_tokens` chunks of `chunk_size_bytes` to the peer worker across the
        cable while the receiver consumes the peer's chunks into a `num_slots`-deep L1 ring, credited
        back through the producer's connection. No input tensors; returns a dummy tensor. Used to
        profile the fabric leg in isolation (inspect Tracy zones).
        )doc",
        &combine_fabric2d,
        nb::arg("device"),
        nb::arg("num_links") = 2,
        nb::arg("num_tokens") = 100,
        nb::arg("chunk_size_bytes") = 14336,
        nb::arg("num_slots") = 32,
        nb::arg("axis") = 0,
        nb::arg("stall_telemetry") = 0,
        nb::arg("variant") = 0,
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
                d["valid"] = w.valid;
                d["tokens_sent"] = w.tokens_sent;
                d["credits_forwarded"] = w.credits_forwarded;
                d["chunk_size_bytes"] = w.chunk_size_bytes;
                d["num_slots"] = w.num_slots;
                d["write_up_to_final"] = w.write_up_to_final;
                d["t_first_send"] = w.t_first_send;
                d["t_last_send"] = w.t_last_send;
                d["t_last_credit"] = w.t_last_credit;
                d["edm_slots"] = w.edm_slots;
                d["credit_packets"] = w.credit_packets;
                d["loop_iters"] = w.loop_iters;
                d["wait_slot_cycles"] = w.wait_slot_cycles;
                d["issue_cycles"] = w.issue_cycles;
                d["starve_cycles"] = w.starve_cycles;
                d["credit_cycles"] = w.credit_cycles;
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
