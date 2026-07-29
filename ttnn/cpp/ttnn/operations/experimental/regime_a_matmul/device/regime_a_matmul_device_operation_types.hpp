// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_config.hpp"

namespace ttnn::experimental::prim {

struct RegimeAMatmulParams {
    // The whole config drives compile-time kernel args, so it lives in operation_attributes and is
    // keyed by the framework's default reflection-based program hash.
    std::optional<RegimeAMatmulConfig> config;

    // ---- Production single-chip fusions (all optional; nullopt/1 => byte-identical no-fusion path). ----
    // Applied at the output/compute stage (post split-K reduction for Pk>1) so no extra output DRAM
    // round-trip. All fusion presence flags participate in the reflection-based program-cache hash.
    std::optional<operations::unary::UnaryWithParam> fused_activation;  // Y = act(A@B + bias)
    // addcmul: Y = residual + scalar*(A@B + bias)*gate. Scalar present <=> addcmul active (residual/gate
    // tensors live in RegimeAMatmulInputs). Rejected together with fused_activation.
    std::optional<float> fused_ternary_scalar;
    int32_t chunks = 1;  // output column-split count (regime_a_matmul_split); 1 => single output tensor

    // ---- TEST-ONLY critical-path ablation diagnostic BITMASK (NOT public API). 0 => production
    // (byte-identical: no kernel defines, no extra runtime args). Each bit compile-gates one stage skip:
    //   bit0 (1)  SKIP_ALL_IN0_READ        - every core skips its in0 DRAM read
    //   bit1 (2)  SKIP_REDUNDANT_IN0_READ  - only ns>0 groups skip their duplicate in0 DRAM reads
    //   bit2 (4)  SKIP_IN0_RING_FORWARD    - skip ring payload writes, keep readiness/credit semaphores
    //   bit3 (8)  SKIP_COMPUTE             - skip the matmul math; keep CB plumbing + minimal output pack
    //   bit4 (16) SKIP_REDUCTION           - skip split-K chain sends/receives + accumulation (write local partial)
    //   bit5 (32) SKIP_OUTPUT_WRITE        - skip output DRAM payload writes; keep iteration + CB consumption
    // Two further bits PERTURB (rather than skip) the ring forward, to attribute its cost between hop
    // distance/link contention and per-core injection/L1-source bandwidth. Both keep the readiness semaphore
    // to the TRUE ring neighbour, so the dependency chain and step count are byte-for-byte the baseline's:
    //   bit6 (64)  FWD_NEAR               - same bytes, but the payload goes to the NEAREST program core on
    //                                       this core's writer NoC instead of the ring successor (~1 hop)
    //   bit7 (128) FWD_HALF               - true destination, half the payload bytes (byte-linearity probe)
    // Bit8 is different in kind: a HOST-ONLY, correctness-preserving alternative in0 ring topology (no
    // kernel define, no extra arg, output still valid), so it is allowed on every path including fusion:
    //   bit8 (256) RING_REGIONAL          - partition the 8*Ns cores of each (kk,mm) group into Ns
    //                                       physically compact rings instead of "the 8 banks of one slice"
    //                                       (MEASURED REFUTED: -4..-9%; compactness raises peak link load)
    //   bit9 (512) RING_BALANCED          - production ring membership, but order each ring to minimise the
    //                                       peak GLOBAL NoC link load instead of that ring's own hop cost
    //                                       (in0-only, unbudgeted: +4.2% / -10% depending on shape)
    //   bit10 (1024) RING_BALANCED_BG      - bit9 + the fixed background traffic (in1 reads, in0 read,
    //                                       reduction, output) on the link map, + worst-edge and hop
    //                                       budgets anchored on production, + adopt-only-if-better gate
    //                                       (CORPUS-REFUTED at deployed configs: 0 wins, -13.7% worst)
    // Back to kernel-behaviour bits (invalid output, unfused/single-output only):
    //   bit11 (2048) SKIP_IN1_READ         - drop the in1 DRAM read payload; keep CB reserve/push, rotated
    //                                       shard order, barriers, M-split forwarding, semaphores, compute
    // And one more HOST-ONLY, correctness-preserving bit (valid output, allowed on every path):
    //   bit13 (8192) PLACE_MESH            - force the 2D (bank x slice) mesh placement ON. It is PRODUCTION
    //                                       DEFAULT when Pk>=10 && Ns==1 && Sm==1 (see the gate in the
    //                                       factory); this bit forces it for shapes outside the gate.
    //   bit14 (16384) MESH_OFF             - force the mesh OFF, i.e. restore the pre-mesh placement, so the
    //                                       shipped default can be A/B'd
    //   bit15 (32768) MESH_SPREAD          - mesh variant: when there are fewer slices than grid rows, space
    //                                       them evenly over all rows instead of packing rows 0..preaders-1
    //   bit16 (65536) SUBBLOCK_LEGACY      - restore the old subblock sizer (subblock_h capped at 2). The
    //                                       default now enlarges an under-4-tile subblock to the largest area
    //                                       that fits the fp32 DST limit; bit-exact, so this bit is for A/B.
    //   bit12 (4096) PLACE_IN1_OPT         - CROSS placement: put each (bank, noc) reader group in the region
    //                                       downstream of THAT endpoint on THAT NoC instead of one spiral
    //                                       around the NOC_0-optimal core (also supersedes IN1_NEAR pass 1)
    // Bits combine freely (pair-interaction matrix). bit0 dominates bit1 (normalize skip-all+redundant to
    // skip-all). Set from TT_REGIME_A_DIAG_MASK in invoke(); part of the reflection program-cache hash so
    // each mask is a distinct cached program. Diagnostic outputs are intentionally invalid; correctness is
    // asserted only for mask 0. Supported only on the unfused / single-output path (factory TT_FATALs else). ----
    uint32_t diag_mask = 0;

    // NOTE: numerics are FIXED production behavior, not options — BF16 in/out, HiFi2, FP32 dest-accumulation,
    // DRAM-interleaved output. There is deliberately no output dtype / memory_config / compute_kernel_config
    // here: they were previously accepted but ignored (an API-correctness hazard), so they are not part of the
    // op's attributes or program-cache identity. The split `dim` is always -1 (validated in the wrapper) and is
    // likewise not stored/hashed — only `chunks` reaches the device op.
};

struct RegimeAMatmulInputs {
    Tensor input_tensor;   // in0 : [.., M, K], DRAM interleaved, bf16, TILE
    Tensor weight_tensor;  // in1 : [.., K, N], DRAM width-sharded (8 banks), bf16, TILE

    // ---- Optional fusion operands (DRAM interleaved, TILE). ----
    std::optional<Tensor> bias_tensor;            // [.., 1, N] / [.., N] row-broadcast bias
    std::optional<Tensor> fused_ternary_input_a;  // addcmul residual/base, full [M, N]
    std::optional<Tensor> fused_ternary_input_b;  // addcmul gate/multiplier, [1, N] bcast or [M, N] full
};

}  // namespace ttnn::experimental::prim
