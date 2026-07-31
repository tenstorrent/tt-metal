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
        MoE prefill combine over an explicitly-forwarded FABRIC_2D: expert-processed tokens go back to
        the chips they came from, chip-local DRAM -> eth -> destination chip's DRAM.

        Called exactly like `ttnn.experimental.deepseek_prefill.combine`, plus one extra tensor and this
        op's tuning knobs. The work is described entirely by control tensors the caller has already
        staged in DRAM; the op discovers it on device.

            dispatched_buffer     the tokens, one per page. A chip's page range for one expert holds
                                  that expert's tokens grouped by the chip they ORIGINATED on.
            dispatched_metadata   3 int32 per token: (linearized_coord, token_idx, topk_idx). The
                                  token's destination slot is page token_idx * num_experts_per_tok +
                                  topk_idx of the origin chip's output.
            expert_token_counts   tokens per expert over all origin chips; closes the last run.
            expert_region_offsets where each expert's region starts in dispatched_buffer.
            expert_offsets        the one tensor the production op does not take: where each ORIGIN
                                  chip's run starts inside each expert's region. Must be REPLICATED
                                  along the dispatch-group axis, since every chip needs every origin
                                  chip's boundaries for the experts it hosts.

        The op owns everything about HOW: which cores run, which cable each drives, how a destination
        more than one hop away is reached (it forwards through the intervening chips' DRAM itself
        rather than asking the fabric to), and how the work splits across producers. Allocates and
        returns the combined output, (1, 1, seq_len_per_chip, num_experts_per_tok, emb_dim) BFLOAT16
        ROW_MAJOR per device — the same shape the production op returns.

        BFLOAT16 ROW_MAJOR input only, and `init_zeros` must be false.
        )doc",
        &combine_fabric2d,
        nb::arg("device"),
        nb::arg("dispatched_buffer"),
        nb::arg("dispatched_metadata"),
        nb::arg("expert_token_counts"),
        nb::arg("expert_region_offsets"),
        nb::arg("expert_offsets"),
        nb::arg("dispatch_group_size"),
        nb::arg("experts_per_chip"),
        nb::arg("num_experts_per_tok"),
        nb::arg("seq_len_per_chip"),
        nb::arg("axis") = 0,
        nb::arg("num_links") = 2,
        nb::arg("topology") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("init_zeros") = false,
        nb::arg("num_l1_slots") = 8,
        nb::arg("fwd_bump_every") = 32,
        nb::arg("assignment_order") = 1,
        nb::arg("stall_telemetry") = 0);

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
