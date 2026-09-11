// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The vsa_sdpa streaming descriptor builder, shared by the single-device streaming factory
// (VsaSdpaStreamProgramFactory, ring == nullptr) and the fused ring op (vsa_ring_sdpa), which passes a
// per-mesh-coordinate ring context. See VSA_RING_SDPA_SPEC.md.

#include <optional>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa/device/vsa_sdpa_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Kernel HANDLES equal descriptor push order (the runtime assigns them sequentially; the experimental ring
// SDPA and the standalone ring all-gather rely on the same). The builder pushes: reader(workers),
// writer(workers), reader(leaders), writer(leaders), compute; in ring mode the all-gather helper's four
// kernels follow (fwd reader, fwd writer, bwd reader, bwd writer). NEVER resolve handles through
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
    ttnn::MeshCoordinate coord{0, 0};
    std::optional<ttnn::MeshCoordinate> forward_coord;
    std::optional<ttnn::MeshCoordinate> backward_coord;
    const ttnn::experimental::prim::RingAttentionAllGatherAsyncParams* ag = nullptr;  // links, topology, semaphores
    const Tensor* gathered_k = nullptr;  // [1,H,T_local*ring_size,d] persistent all-gather buffers
    const Tensor* gathered_v = nullptr;
    tt::tt_metal::CoreCoord ccl_core_grid_offset{0, 0};  // must be (grid.x-1, 0): the senders' column
};

tt::tt_metal::ProgramDescriptor build_vsa_sdpa_stream_descriptor(
    const VsaSdpaParams& attrs, const VsaSdpaInputs& t, Tensor& output, const VsaRingContext* ring);

// Cache-hit re-application of the VSA kernels' raw uint32 address args (dense mask, stream order) and
// buffer addresses. `ring` is informational (the patch walks each kernel instance's own per-core table).
void patch_vsa_sdpa_stream_runtime_args(
    tt::tt_metal::Program& program,
    const VsaSdpaParams& attrs,
    const VsaSdpaInputs& t,
    Tensor& tensor_return_value,
    bool ring);

}  // namespace ttnn::prim
