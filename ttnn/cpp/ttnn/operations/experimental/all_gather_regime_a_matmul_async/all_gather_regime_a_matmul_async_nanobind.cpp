// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_regime_a_matmul_async_nanobind.hpp"

#include <optional>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "all_gather_regime_a_matmul_async.hpp"
#include "device/all_gather_regime_a_matmul_async_plan.hpp"
#include "device/all_gather_regime_a_matmul_async_device_operation_types.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"

namespace pl = ttnn::operations::experimental::agmm::plan;

namespace ttnn::operations::experimental::all_gather_regime_a_matmul_async::detail {

void bind_all_gather_regime_a_matmul_async(nb::module_& mod) {
    ttnn::bind_function<"all_gather_regime_a_matmul_async", "ttnn.experimental.">(
        mod,
        R"doc(
        all_gather_regime_a_matmul_async(input_tensor, weight_tensor, config=None, cluster_axis=None, topology=ttnn.Topology.Ring, num_links=1, num_workers_per_link=1, num_buffers_per_channel=2, multi_device_global_semaphore=[], barrier_semaphore=None, persistent_output_buffer=None, memory_config=None, dtype=None, compute_kernel_config=None)

        Fused all-gather(in0, dim=-1) @ in1 using regime_a_matmul as the compute engine. in0 owns a
        contiguous K-shard [.., M, K_local]; in1 is the full [.., K_global, N] regime_a DRAM width-shard.
        D (= K_global / K_local) devices are gathered. bf16 only, no transpose/batching, tile-aligned K
        sharding, no epilogues (v1).

        Task-2 scope: D=1 is behaviorally identical to regime_a_matmul; D>1 validates the host plan and
        reports that the fabric-streaming path is implemented in Task 3.
        )doc",
        &ttnn::experimental::all_gather_regime_a_matmul_async,
        nb::arg("input_tensor"),
        nb::arg("weight_tensor"),
        nb::arg("config") = nb::none(),
        nb::kw_only(),
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("topology") = ttnn::ccl::Topology::Ring,
        nb::arg("num_links") = 1,
        nb::arg("num_workers_per_link") = 1,
        nb::arg("num_buffers_per_channel") = 2,
        nb::arg("multi_device_global_semaphore") = std::vector<GlobalSemaphore>{},
        nb::arg("barrier_semaphore") = nb::none(),
        nb::arg("persistent_output_buffer") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());

    // Device-free host-plan preview for offline tests (D=1/2/4/8, K ownership, core collisions, L1 sizing).
    mod.def(
        "all_gather_regime_a_matmul_plan",
        [](uint32_t M,
           uint32_t K,
           uint32_t N,
           uint32_t D,
           const std::string& topology,
           uint32_t num_links,
           uint32_t num_workers_per_link,
           uint32_t Ns,
           uint32_t Pk,
           uint32_t Sm,
           uint32_t kb,
           uint32_t nsb,
           uint32_t C,
           uint32_t transport_slots,
           uint32_t grid_x,
           uint32_t grid_y) {
            pl::AgmmPlanConfig c;
            c.M = M;
            c.K = K;
            c.N = N;
            c.D = D;
            c.topology = (topology == "ring") ? pl::Topology::Ring : pl::Topology::Linear;
            c.num_links = num_links;
            c.num_workers_per_link = num_workers_per_link;
            c.Ns = Ns;
            c.Pk = Pk;
            c.Sm = Sm;
            c.kb = kb;
            c.nsb = nsb;
            c.C = C;
            c.transport_slots = transport_slots;
            c.grid_x = grid_x;
            c.grid_y = grid_y;
            const auto p = pl::build_plan(c);
            nb::dict d;
            d["valid"] = p.valid;
            d["errors"] = p.errors;
            d["Mt"] = p.Mt;
            d["Kt"] = p.Kt;
            d["Nt"] = p.Nt;
            d["global_k_blocks"] = p.global_k_blocks;
            d["k_blocks_per_device"] = p.k_blocks_per_device;
            nb::list devs;
            for (const auto& dp : p.devices) {
                nb::dict dd;
                dd["device_index"] = dp.device_index;
                dd["local_k_blocks"] = dp.local_k_blocks;
                devs.append(dd);
            }
            d["devices"] = devs;
            d["regime_a_cores"] = p.regime_a_cores;
            d["mux_cores"] = p.mux_cores;
            d["fabric_worker_cores"] = p.fabric_worker_cores;
            d["reserved_fabric_cores"] = p.reserved_fabric_cores;
            d["total_cores"] = p.total_cores;
            d["usable_cores"] = p.usable_cores;
            d["core_fit"] = p.core_fit;
            d["core_collision"] = p.core_collision;
            d["transport_chunk_tiles"] = p.transport_chunk_tiles;
            d["transport_chunk_bytes"] = p.transport_chunk_bytes;
            d["transport_l1_bytes"] = p.transport_l1_bytes;
            d["l1_fit"] = p.l1_fit;
            return d;
        },
        nb::arg("M"),
        nb::arg("K"),
        nb::arg("N"),
        nb::arg("D"),
        nb::arg("topology") = "ring",
        nb::arg("num_links") = 1,
        nb::arg("num_workers_per_link") = 1,
        nb::arg("Ns") = 1,
        nb::arg("Pk") = 1,
        nb::arg("Sm") = 1,
        nb::arg("kb") = 1,
        nb::arg("nsb") = 0,
        nb::arg("C") = 1,
        nb::arg("transport_slots") = 2,
        nb::arg("grid_x") = 12,
        nb::arg("grid_y") = 10,
        "Device-free host-plan preview for the fused all-gather + regime_a_matmul op (Task 2).");
}

}  // namespace ttnn::operations::experimental::all_gather_regime_a_matmul_async::detail
