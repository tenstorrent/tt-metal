// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_config.hpp"

namespace ttnn::experimental::prim {

struct AllGatherRegimeAMatmulAsyncParams {
    // The whole config drives compile-time kernel args, so it lives in operation_attributes and is
    // keyed by the framework's default reflection-based program hash.
    std::optional<RegimeAMatmulConfig> config;

    // ---- Production single-chip fusions (all optional; nullopt/1 => byte-identical no-fusion path). ----
    // Applied at the output/compute stage (post split-K reduction for Pk>1) so no extra output DRAM
    // round-trip. All fusion presence flags participate in the reflection-based program-cache hash.
    std::optional<operations::unary::UnaryWithParam> fused_activation;  // Y = act(A@B + bias)
    // addcmul: Y = residual + scalar*(A@B + bias)*gate. Scalar present <=> addcmul active (residual/gate
    // tensors live in AllGatherRegimeAMatmulAsyncInputs). Rejected together with fused_activation.
    std::optional<float> fused_ternary_scalar;
    int32_t chunks =
        1;  // output column-split count (all_gather_regime_a_matmul_async_split); 1 => single output tensor

    // ---- Fused fabric all-gather (PHASE 1, DRAM-staged). ----
    // tp == 1 means "no fabric gather": in0 already holds the full K and every field below is ignored,
    // which is the single-chip path and stays byte-identical to regime_a_matmul. tp >= 2 (even) enables
    // the fused gather: in0 is this device's [M, K/tp] shard and the op gathers the full K itself.
    //
    // Layout contract: the gathered K is laid out CONTIGUOUSLY BY SOURCE RANK -- rank d's shard occupies
    // K offset d * (K/tp). That matches all_gather(dim=-1) exactly, so the fused path is a drop-in for the
    // Phase-0 composition and the in1 reader's rotated-read order is unchanged. The design spec notes a
    // contiguous global-K -> Pk assignment can leave Pk groups idle during fabric startup; that is a
    // deferred PERFORMANCE tweak, not a correctness one.
    //
    // These are program-cache identity: they all feed compile-time kernel args.
    uint32_t tp = 1;               // TP group size along cluster_axis (1 = disabled, else even and >= 2)
    uint32_t cluster_axis = 0;     // mesh axis the TP group runs along
    uint32_t num_links = 1;        // fabric links per direction
    bool topology_is_ring = true;  // ring vs linear device topology

    // Cross-chip arrival credits, supplied by the caller. REQUIRED when tp >= 2 (>= 1 entry; the second is
    // used by the backward stream once it is driven), ignored when tp == 1.
    //
    // These deliberately are NOT CreateSemaphore program semaphores. A program semaphore is zeroed as part
    // of program launch, and a fabric atomic-inc crosses chips with no such ordering: the sending chip can
    // credit us before our own program has launched and written the zero, which erases the credit and hangs
    // the receiver on a count that will never arrive. A global semaphore is allocated up-front by the caller
    // and is live before any device in the group starts, which closes that window.
    //
    // The caller must ping-pong them (depth >= 2) and ttnn.synchronize_device() after creating them; see
    // models/tt_dit/parallel/manager.py. The kernel resets its slot to 0 once it has seen all tp-1 arrivals
    // -- safe because in a store-and-forward ring exactly one neighbour ever increments a given slot, so the
    // (tp-1)'th credit proves that neighbour is done for this invocation.
    std::vector<tt::tt_metal::GlobalSemaphore> gather_semaphores;

    // NOTE: numerics are FIXED production behavior, not options — BF16 in/out, HiFi2, FP32 dest-accumulation,
    // DRAM-interleaved output. There is deliberately no output dtype / memory_config / compute_kernel_config
    // here: they were previously accepted but ignored (an API-correctness hazard), so they are not part of the
    // op's attributes or program-cache identity. The split `dim` is always -1 (validated in the wrapper) and is
    // likewise not stored/hashed — only `chunks` reaches the device op.
};

struct AllGatherRegimeAMatmulAsyncInputs {
    // tp == 1 : in0 is the full [.., M, K].
    // tp >= 2 : in0 is THIS DEVICE'S shard [.., M, K/tp]; the op gathers the rest over fabric into
    //           `gather_staging_buffer`, which then serves as the full-K activation for the matmul.
    Tensor input_tensor;   // in0 : DRAM interleaved, bf16, TILE
    Tensor weight_tensor;  // in1 : [.., K, N], DRAM width-sharded (8 banks), bf16, TILE

    // Fused-gather staging target [.., M, K] (the caller's persistent_output_buffer). Remote ranks write
    // their shards directly into this buffer over fabric; the local shard is copied in at rank offset.
    // REQUIRED when tp >= 2, ignored when tp == 1. Must be double-buffered and synchronised by the caller
    // (see models/tt_dit/parallel/manager.py) -- a single un-synced buffer yields silent partial corruption.
    std::optional<Tensor> gather_staging_buffer;

    // ---- Optional fusion operands (DRAM interleaved, TILE). ----
    std::optional<Tensor> bias_tensor;            // [.., 1, N] / [.., N] row-broadcast bias
    std::optional<Tensor> fused_ternary_input_a;  // addcmul residual/base, full [M, N]
    std::optional<Tensor> fused_ternary_input_b;  // addcmul gate/multiplier, [1, N] bcast or [M, N] full
};

}  // namespace ttnn::experimental::prim
