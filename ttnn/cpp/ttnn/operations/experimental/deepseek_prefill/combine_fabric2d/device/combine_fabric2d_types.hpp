// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// Isolated fabric-transfer experiment op. No input tensors — the op reserves its own L1 scratch.
//
// Every fabric eth core of every device (num_links toward each axis neighbor) gets ONE worker core in
// its physical column, running a producer (writer RISC) and a receiver (reader RISC). Each link is
// full duplex for payload: the producer sends `num_tokens` chunks of `chunk_size_bytes` to the peer
// worker across the cable, while the receiver consumes the peer's chunks. The producer owns the eth
// channel's single fabric connection and also forwards the receiver's credit returns.
//
// `device` is FIRST so the framework's get_first_object_of_type<MeshDevice*>() over
// attribute_values() (tuple element 0) finds the mesh.
struct CombineFabric2dParams {
    ttnn::MeshDevice* device = nullptr;
    uint32_t num_links = 2;
    uint32_t num_tokens = 100;
    uint32_t chunk_size_bytes = 14336;  // 7168 bf16 elements = one token
    uint32_t num_slots = 32;            // ring depth; also the producer's initial write_up_to credit
    uint32_t axis = 0;                  // mesh axis along which the neighbors are chosen
    // Fine-grained stall attribution in the producer (wait-for-slot / issue / credit-starved / credit
    // buckets). Off by default: it costs ~3 wall-clock register reads per token, which is a few percent
    // of the very number being measured. Turn it on to explain a result, off to quote one.
    uint32_t stall_telemetry = 0;
    // Diagnostic overrides for the producer loop. Both deliberately break a guarantee, to price it;
    // neither belongs in a real run.
    //   bit2 (4) RELAXED_READY    data-ready atomic-inc without the flush ordering, so the receiver may
    //                             observe the flag before the payload has landed (worth ~1%)
    //   bit3 (8) NO_FLOW_CONTROL  ignore the credit gate and send no credit packets at all, which
    //                             overwrites the receiver ring in flight but measures the payload-only
    //                             ceiling of the link (23.8 GB/s per direction on 8x4 BH)
    // Phase 3 receiver-path modes (mutually exclusive; pick at most one). Both make the op land tokens
    // in a real interleaved DRAM output buffer instead of NOP-acking them in L1.
    //   bit5 (32) DRAM_DIRECT  Approach #1: producer writes each token straight to the peer chip's DRAM
    //                          (page base + token), no credits/semaphores, no receiver kernel. Plain
    //                          unicast write, not the fused write+atomic-inc (that hangs BH on DRAM).
    //   bit6 (64) DRAM_DRAIN   Approach #2: unchanged fabric path into the L1 ring; the receiver then
    //                          writes each consumed slot to DRAM over NOC_0 and waits for it to land
    //                          before returning that slot's credit.
    uint32_t variant = 0;
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Mesh;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "device",
        "num_links",
        "num_tokens",
        "chunk_size_bytes",
        "num_slots",
        "axis",
        "stall_telemetry",
        "variant",
        "topology");
    auto attribute_values() const {
        return std::forward_as_tuple(
            device, num_links, num_tokens, chunk_size_bytes, num_slots, axis, stall_telemetry, variant, topology);
    }
};

struct CombineFabric2dInputs {};

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
