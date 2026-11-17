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

struct RelayAffinePrefixParams {
    uint32_t sequence_parallel_axis;
    uint32_t local_rows;
    uint32_t num_links;
    // The same number of cores per head, head by head (core j of a head corrects its j-th share of the chunks, and the
    // head's first core also runs the relay's hop), then two sender cores per link: the relay's, and the final carry's
    // second.
    std::vector<tt::tt_metal::CoreCoord> cores;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    // L1 addresses of the caller's entry-state, final-state and final-carry arrival counters, kept alive across
    // calls: a peer counts into them before this rank's call starts, and each call subtracts what it consumed.
    std::array<uint32_t, 3> semaphore_addresses{};
};

// The landing buffer's K x K slots per head core: entry state, final state.
inline constexpr uint32_t relay_landing_slots = 2;

struct RelayAffinePrefixInputs {
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
    // The rank's final transition in complement form, which the first rank applies to the relayed final state when
    // the interval ends in a separated tail, and the optional exclusive valid end that decides it.
    Tensor final_e;
    Tensor final_b;
    std::optional<Tensor> actual_end;
};

}  // namespace ttnn::experimental::prim
