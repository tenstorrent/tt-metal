// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/dram_sender_global_circular_buffer.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace ttnn {

// Bench-only companion to `ttnn.dram_prefetcher`: enqueues a Program that loads a
// discard-only kernel on each receiver core of the supplied GCB. Each receiver
// runs `wait_front(1); pop_front(1);` in a loop `num_iters` times, draining
// pages pushed by the prefetcher and throwing them away. Used to measure the
// prefetcher's push bandwidth without matmul receiver-side effects.
//
// Exactly one of `global_cb` / `dram_sender_global_cb` must be set, matching
// whichever GCB the sender was given.
void dram_prefetcher_consumer(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    uint32_t num_iters,
    uint32_t page_size_bytes,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb = std::nullopt,
    const std::optional<const tt::tt_metal::experimental::DramSenderGlobalCircularBuffer>& dram_sender_global_cb =
        std::nullopt);

}  // namespace ttnn
