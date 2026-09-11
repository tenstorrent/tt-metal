// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The vsa_sdpa streaming descriptor builder, shared by the single-device streaming factory
// (VsaSdpaStreamProgramFactory, ring == nullptr) and the fused ring op (vsa_ring_sdpa), which passes a
// per-mesh-coordinate ring context. See VSA_RING_SDPA_SPEC.md.

#include <optional>
#include <vector>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/transformer/sdpa/device/vsa_sdpa_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Kernel HANDLES equal descriptor push order (the runtime assigns them sequentially; the experimental ring
// SDPA and the standalone ring all-gather rely on the same). The builder pushes: reader(workers),
// writer(workers), reader(leaders), writer(leaders), compute; in ring mode the all-gather's reader and
// writer follow (added to the materialized Program by the ring factory). NEVER resolve handles through
// collect_kernel_meta: it iterates an unordered map, and using its index as a handle wrote the VSA reader's
// buffer addresses into the all-gather writer's NoC-coordinate slots on every cache hit (ring deadlock).
inline constexpr uint32_t kVsaReaderWorkerKernel = 0;
inline constexpr uint32_t kVsaWriterWorkerKernel = 1;
inline constexpr uint32_t kVsaReaderLeaderKernel = 2;
inline constexpr uint32_t kVsaWriterLeaderKernel = 3;
inline constexpr uint32_t kVsaComputeKernel = 4;
inline constexpr uint32_t kVsaStreamKernelCount = 5;

struct VsaRingContext {
    uint32_t device_index = 0;  // this device's position (= SP shard) on the ring axis
    uint32_t ring_size = 0;
    uint32_t forward_writes_expected = 0;   // shards arriving over the forward chain
    uint32_t backward_writes_expected = 0;  // shards arriving over the backward chain
    uint32_t sender_rows = 1;               // grid rows (from row 0) reserved for the all-gather sender cores
    uint32_t workers_per_direction = 0;     // all-gather workers per direction (links x workers/link): poll table size
    const Tensor* gathered_kv = nullptr;    // [1,1,T_local*ring_size,2*H*d] persistent all-gather buffer (flat K|V)
    // Filled by the builder: the leaders' NoC coords and the two program-semaphore ids the all-gather signals.
    std::vector<tt::tt_metal::CoreCoord> receiver_cores_noc;
    std::vector<uint32_t> receiver_semaphores;
};

tt::tt_metal::ProgramDescriptor build_vsa_sdpa_stream_descriptor(
    const VsaSdpaParams& attrs, const VsaSdpaInputs& t, Tensor& output, VsaRingContext* ring);

// Common-runtime-arg layout of the ring leaders (reader and writer), after the accessor common args. Entries
// marked * are raw addresses the ring factory re-applies on cache hits; [7], [8], [10..] are filled by the ring
// factory once the all-gather has placed its cores (the descriptor emits zeros of the right size).
//   [0]  gathered K/V buffer address *
//   [1]  ring_index                 [2] blocks_per_shard          [3] Wt: tiles per flat K/V row (2*H*d/32)
//   [4]  DHt (d/32)                 [5] H                         [6] Skt (block_size/32)
//   [7]  tiles per fabric packet    [8] packets per landed-count increment (the all-gather's chunks_per_sync)
//   [9]  G: all-gather workers per direction
//   [10] out_ready_sem L1 address on the direction-0 workers *    [11] the direction-1 workers' *
//   [12 + 3*(d*G + g) + {0, 1, 2}]  worker g of direction d: NoC x | y << 16, first tile, end tile of its slice range
//   [12 + 6*G ..]  the 9 RingSDPAOpReceiver words (reader only)
inline constexpr uint32_t kRingCommonArgGatheredAddr = 0;
inline constexpr uint32_t kRingCommonArgTilesPerPacket = 7;
inline constexpr uint32_t kRingCommonArgChunksPerSync = 8;
inline constexpr uint32_t kRingCommonArgSemAddr0 = 10;
inline constexpr uint32_t kRingCommonArgSemAddr1 = 11;
inline constexpr uint32_t kRingCommonArgPollTable = 12;
inline constexpr uint32_t kRingPollWordsPerWorker = 3;

// Cache-hit re-application of the VSA kernels' raw uint32 address args (dense mask, stream order) and
// buffer addresses. `ring` is informational (the patch walks each kernel instance's own per-core table).
void patch_vsa_sdpa_stream_runtime_args(
    tt::tt_metal::Program& program,
    const VsaSdpaParams& attrs,
    const VsaSdpaInputs& t,
    Tensor& tensor_return_value,
    bool ring);

}  // namespace ttnn::prim
