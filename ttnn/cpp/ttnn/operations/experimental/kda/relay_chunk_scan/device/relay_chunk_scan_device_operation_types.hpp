// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// The relay's persistent counters: its entry-state, final-state and final-carry arrivals, then the output gate's
// handshakes: x's and W's block counts, and a head core's g landed.
inline constexpr uint32_t relay_semaphores = 6;

struct RelayChunkScanParams {
    uint32_t sequence_parallel_axis;
    uint32_t local_rows;
    uint32_t num_links;
    // The same number of cores per head, head by head (core j of a head corrects its j-th share of the chunks, and the
    // head's first core also runs the relay's hop), then two sender cores per link: the relay's, and the final carry's
    // second, then a helper core per head (which forms the output gate of the head's first core's chunks).
    std::vector<tt::tt_metal::CoreCoord> cores;
    // The gated output norm: heads per batch (the output is [B, T, H*V]), the output gate's math fidelity, epsilon,
    // and the output's dtype and memory.
    uint32_t num_heads;
    tt::tt_metal::MathFidelity gate_math_fidelity;
    float norm_epsilon;
    tt::tt_metal::DataType output_dtype;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    // L1 addresses of the caller's relay_semaphores counters, kept alive across calls (each call subtracts what it
    // consumed): a peer counts into the first three before this rank's call starts.
    std::array<uint32_t, relay_semaphores> semaphore_addresses{};
    uint32_t slab_index_scale = 1;
    uint32_t slab_index_offset = 0;
};

// The landing buffer's K x K slots per head core: entry state, final state.
inline constexpr uint32_t relay_landing_slots = 2;

struct RelayChunkScanInputs {
    Tensor packed;
    // Per chunk, the output for a zero seed F(0) and an identity seed F(I): height-sharded so each core's F(0) rows,
    // then its F(I) rows, are in its own L1, or interleaved [B*H, 2, local rows, K] and streamed.
    Tensor terms;
    Tensor initial_state;
    Tensor actual_start;
    // L1, height-sharded relay_landing_slots K x K states per head core (each head's first core), in order: the entry
    // state the predecessor writes and the final state the last rank writes into the first rank's. Consecutive calls
    // need a barrier across the SP line between them: a rank's predecessor's next call overwrites them as soon as it
    // starts.
    Tensor landing;
    // L1, height-sharded over the heads' cores and the helpers, the same bytes on each: the output gate's K blocks of x
    // and W (and the head-only buffers they alias). Only used within the call: allocated right before it, it lands in
    // whatever L1 the layer has freed by then.
    Tensor gate_workspace;
    // The rank's final transition in complement form, which the first rank applies to the relayed final state when
    // the interval ends in a separated tail, and the optional exclusive valid end that decides it.
    Tensor final_e;
    Tensor final_b;
    std::optional<Tensor> actual_end;
    // The gated output norm: its gate x @ W from the [B, T, X] input x and the [X, H*V] weight W, and the [V] weight.
    // With gate_local, x's columns of this rank's TP slice are read from gate_local's [B, T, X / TP] instead.
    Tensor gate_input;
    std::optional<Tensor> gate_local;
    Tensor gate_weight;
    Tensor norm_weight;
    // initial_state is then a [entries, H, K, K] slab, read at its entry (zeros at actual_start == 0); the final carry
    // is written back into the same entry, in place, and the slab is returned as the carry.
    std::optional<Tensor> slab_index;
};

// The heads and K of a relay: initial_state is [B*H, K, K], or a [entries, H, K, K] slab with slab_index.
struct RelayStateGeometry {
    uint32_t heads;
    uint32_t key_dim;
};
inline RelayStateGeometry relay_state_geometry(const RelayChunkScanInputs& in) {
    const auto& state = in.initial_state.logical_shape();
    return in.slab_index ? RelayStateGeometry{state[1], state[2]} : RelayStateGeometry{state[0], state[1]};
}

}  // namespace ttnn::experimental::prim
