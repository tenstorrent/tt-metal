// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host-side helpers shared by the sequence-parallel (dim-2) matmul + collective fusions:
//   * all_gather_matmul_sp_async  (gather-then-multiply, MatmulFusedOpSignalerType::SP_ALL_GATHER)
//   * matmul_reduce_scatter_sp_async (multiply-then-scatter, MatmulFusedOpSignalerType::SP_REDUCE_SCATTER)
//
// Mechanism (see the 2D mcast matmul factory, SP_SLICE_SCHEDULE): a dim-2 slice of [B,1,S,X] is a contiguous block of
// S/T tile-rows inside each batch, so viewing the tensor as [B*T,1,S/T,X] makes matmul sub-batch b*T+t exactly slice
// t of batch b. With fuse_batch=false the 2D mcast matmul already iterates sub-batches with stride MtKt / MtNt; the SP
// signaler only replaces the implicit 0..B*T-1 order by a caller-provided schedule. Nothing in here is topology
// specific: the AG/RS-order derivations live with the fused ops.

#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::ccl {

// One matmul batch-loop iteration of an SP schedule.
struct SpSubBatch {
    uint32_t in0_idx = 0;     // sub-batch index used for the in0 tile offset (in0_core_offset + in0_idx*MtKt)
    uint32_t out_idx = 0;     // sub-batch index used for the output tile offset (out_core_offset + out_idx*MtNt)
    uint32_t wait_dir = 0;    // SP_ALL_GATHER remote slices: direction semaphore to wait on (0/1)
    uint32_t wait_count = 0;  // SP_ALL_GATHER remote slices: wait_min value on that semaphore (1-based ordinal)
    bool is_local = false;    // SP_ALL_GATHER: read in0 from the original sharded input (no wait)
};

// Metadata-only reshape [B,1,S,X] -> [B*num_slices,1,S/num_slices,X] (same device buffer). TT_FATALs on any shape that
// would need data movement (rank != 4, dim 1 != 1, S not a multiple of num_slices*tile_height, not TILE, not on device).
Tensor sub_batched_view(const Tensor& t, uint32_t num_slices);

// Derived 2D-mcast program config for one sub-batched matmul (no tuning table): fuse_batch=false,
// per_core_M = ceil(Mt_slice / grid.y), per_core_N = ceil(Nt / grid.x), in0_block_w = largest divisor of Kt that is
// <= 4, out blocks shrunk (largest area first) until the CBs fit L1, subblocks from get_matmul_subblock_params.
// `in0_view` is the [B*T,1,S/T,K] view produced by sub_batched_view (Mt is taken from ONE sub-batch).
operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig sp_matmul_program_config(
    const Tensor& in0_view,
    const Tensor& in1,
    tt::tt_metal::CoreCoord grid,
    bool transpose_b,
    const DeviceComputeKernelConfig& compute_kernel_config);

// Packs a schedule into the kernel words consumed under SP_SLICE_SCHEDULE:
//   in0_idx | out_idx << 8 | wait_dir << 16 | is_local << 17 | wait_count << 24
// TT_FATALs on any field that does not fit its bit range.
std::vector<uint32_t> pack_sp_schedule(const std::vector<SpSubBatch>& order);

// RS-side companion of a schedule: ordinal[out_idx] = position of that sub-batch in the matmul schedule, for all
// B*T sub-batches (the inverse permutation of out_idx). TT_FATALs unless out_idx is a permutation of 0..B*T-1.
std::vector<uint32_t> sp_rs_ordinals(const std::vector<SpSubBatch>& order, uint32_t B, uint32_t T);

}  // namespace ttnn::experimental::ccl
