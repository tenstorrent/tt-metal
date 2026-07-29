// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "combine_fabric2d_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// ---------------------------------------------------------------------------------------------
// Telemetry readback
//
// Each producer stamps a small record into a fixed 1 kB L1 region so bandwidth can be recovered
// AFTER the op has run, without re-running under the profiler. `magic` is written last (and zeroed at
// kernel entry), so a partial/absent record is detectable rather than read as stale garbage.
//
// The host side recomputes placement to know which cores to read — placement is a pure function of
// (device, axis, num_links), so it does not need anything retained from the op's run. That is also
// why the caller does NOT need to know where the worker cores are.
// ---------------------------------------------------------------------------------------------
struct CombineFabric2dWorkerTelemetry {
    // Identity, filled in by the host from the recomputed placement.
    uint32_t device_id = 0;
    std::vector<uint32_t> mesh_coord;
    CoreCoord worker_logical;
    CoreCoord worker_physical;
    CoreCoord eth_logical;
    uint32_t eth_phys_x = 0;
    uint32_t link_idx = 0;
    bool relocated = false;  // worker is not in its eth core's physical column
    uint32_t peer_mesh_id = 0;
    uint32_t peer_chip_id = 0;
    // Mesh coordinate of the chip this worker's tokens land on (the far end of its cable). Empty only
    // if that chip is somehow outside this mesh, which the op itself would already have rejected.
    std::vector<uint32_t> peer_coord;

    // Payload, read out of L1. `valid` is false when the magic word is absent (kernel never ran, or
    // died before completing its record).
    bool valid = false;
    uint32_t tokens_sent = 0;
    uint32_t token_size_bytes = 0;
    uint32_t num_in_tokens = 0;
    uint64_t t_first_send = 0;  // wall clock when the first token was handed to the fabric
    uint64_t t_last_send = 0;   // ... the last token
    uint64_t t_drained = 0;     // ... when the EDM drain proved every payload packet reached the far chip.
                                // An upper bound on the transfer where t_last_send is the lower one.

    // How many header-only fillers the drain needed (= edm_slots - 1), and where this producer wrote.
    uint32_t drain_packets = 0;
    uint32_t out_base_page = 0;  // first page written in the PEER chip's output region

    // Stall attribution over the send window. The two cycle buckets are disjoint and together with the
    // loop's own overhead account for t_last_send - t_first_send, which is what makes them useful for
    // deciding whether the ceiling is the eth side or our own issue cost.
    uint32_t edm_slots = 0;         // EDM sender-channel depth (packets in flight it can absorb)
    uint64_t wait_slot_cycles = 0;  // waiting for an EDM slot => eth/eRISC-limited
    uint64_t issue_cycles = 0;      // building + issuing a payload packet
};

struct CombineFabric2dTelemetry {
    uint32_t clock_mhz = 0;  // device AICLK, for turning cycle deltas into time
    std::vector<CombineFabric2dWorkerTelemetry> workers;
};

// Read every worker's telemetry record across the whole mesh. Must be called with the same
// `num_links`/`axis` the op ran with, otherwise placement (and hence which cores are read) differs.
CombineFabric2dTelemetry read_telemetry(ttnn::MeshDevice* mesh_device, uint32_t num_links, uint32_t axis);

struct CombineFabric2dProgramFactory {
    // Contract-2 declarative WorkloadDescriptor entry point. Builds one ProgramDescriptor per mesh
    // coordinate: each chip sends to its own neighbor, so the compile-time args are coord-dependent and
    // cannot be replicated. No workload-scope semaphores are needed — the op has no application-level
    // credit loop, only the EDM's own sender-slot backpressure.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const CombineFabric2dParams& operation_attributes,
        const CombineFabric2dInputs& tensor_args,
        ttnn::Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
