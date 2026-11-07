// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/global_semaphore.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// Fused TP all-gather + matmul: every device holds a K shard A[g] [1, 1, M, K / ranks] of in0 along cluster_axis and
// the whole weight W [1, 1, K, N]; the output is the gathered A times W, [1, 1, M, N], on every device.
//
// The matmul runs program_config (a 2D multicast schedule) on its grid, which must leave the worker column right of it
// to the gather. It starts with this device's own K slice, read from its shard, and takes each other device's slice
// once the gather has landed it in `gathered`, so the gather runs beside the matmul rather than before it.
//
//   gathered          persistent [1, 1, M, K] BFLOAT16 TILE DRAM interleaved buffer the op fills with the gathered A
//                     (also this device's own slice); valid after the call, for other readers of the gathered A
//   ready_semaphore   persistent global semaphore on the gather column's cores, zero before the first call
//   slice_counters    ranks - 1 persistent global semaphores on the matmul's first column (its in0 senders), zero
//                     before the first call; every call leaves them zero
//   num_links         fabric links per direction the gather uses (two senders per link)
Tensor all_gather_matmul_prefill(
    const Tensor& input_tensor,
    const Tensor& weight,
    const Tensor& gathered,
    const tt::tt_metal::GlobalSemaphore& ready_semaphore,
    const std::vector<tt::tt_metal::GlobalSemaphore>& slice_counters,
    int32_t cluster_axis,
    const operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig& program_config,
    uint32_t num_links = 1,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DataType>& dtype = std::nullopt);

}  // namespace ttnn::experimental
