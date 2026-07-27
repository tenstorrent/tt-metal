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
    // Producer loop variant bitmask, so alternatives can be A/B'd without re-plumbing the op:
    //   bit0 (1)  BATCH_CREDITS   forward credits only once a batch has built up (or when blocked)
    //   bit1 (2)  SLOT_HEADERS    one prebuilt packet header per ring slot, non-blocking header send
    //   bit2 (4)  RELAXED_READY   data-ready atomic-inc without the flush ordering (DIAGNOSTIC: the
    //                             receiver may observe the flag before the payload has landed)
    //   bit3 (8)  NO_FLOW_CONTROL producer ignores the credit gate (DIAGNOSTIC: prices the gate; the
    //                             receiver ring is overwritten in flight)
    //   bit4 (16) SINGLE_SRC      producer sends the same source chunk every token instead of rotating
    //                             through num_slots of them. The payload is garbage either way, and it
    //                             frees the L1 that a deep receiver ring needs.
    // Default = BATCH_CREDITS | SLOT_HEADERS: the two changes that are not harness-specific, worth
    // +62% together (13.2 -> 21.3 GB/s per direction). 0 restores the original loop for A/B runs.
    uint32_t variant = 3;
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
