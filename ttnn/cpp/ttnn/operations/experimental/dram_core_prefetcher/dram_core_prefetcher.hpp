// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/global_circular_buffer.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::experimental {

// Thin ttnn-side wrappers around tt::tt_metal::experimental::Start/StopDramCorePrefetcher.
// `tensors` is the same list shape as ttnn::dram_prefetcher: data tensors followed by a
// trailing tensor_addrs tensor (unused on the DRAM-core path but kept for shape parity).
//
// start returns immediately; the kernel runs async on its DRISC core(s). Callers enqueue
// the consuming matmul programs after start, then call stop to drain.
void start_dram_core_prefetcher(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const std::vector<ttnn::Tensor>& tensors,
    uint32_t num_layers,
    const GlobalCircularBuffer& global_cb,
    bool enable_performance_mode = false);

void stop_dram_core_prefetcher(tt::tt_metal::distributed::MeshDevice* mesh_device);

}  // namespace ttnn::operations::experimental
