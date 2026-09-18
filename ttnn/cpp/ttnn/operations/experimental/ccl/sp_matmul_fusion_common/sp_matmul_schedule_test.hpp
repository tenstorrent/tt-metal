// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Test-only entry point: runs the 2D-mcast matmul with a MatmulFusedOpSignaler carrying a caller-provided schedule
// and no CCL, so the SP_SLICE_SCHEDULE kernel path can be checked bitwise against a plain ttnn::matmul with the same
// program config.
//   ag_mode=false: SP_REDUCE_SCATTER. The per-batch barrier+signal of the RS path targets a dummy semaphore on one
//                  core outside the matmul grid; wait/local bits must be 0.
//   ag_mode=true:  SP_ALL_GATHER with in0_alt_addr = the input itself, so is_local iterations read the same data
//                  through the alternate accessor; wait_count must be 0 (wait_min(0) is trivially satisfied), so the
//                  SP_AG_WAIT path runs end to end without an all-gather.

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// in0_view: [B*T,1,Ms,K] (see sp_matmul_fusion_common::sub_batched_view), in1: [1,1,K,N] or [1,1,N,K] (transpose_b).
// schedule_words: B*T packed words (pack_sp_schedule); in0_idx/out_idx must each be a permutation of 0..B*T-1.
// Returns [B*T,1,Ms,N].
Tensor sp_matmul_schedule_test(
    const Tensor& in0_view,
    const Tensor& in1,
    const std::vector<uint32_t>& schedule_words,
    bool transpose_b,
    const operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    bool ag_mode = false);

}  // namespace ttnn::experimental
