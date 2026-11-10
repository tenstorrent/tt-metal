// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The experts' GLU activation act(gate, up), shared by the shared expert (moe_head shared_expert_compute.cpp) and the
// routed experts (flat_routed_expert gate_up_compute.cpp). It runs on the PACK thread's SFPU, between the matmul and
// the pack, over a (gate, up) pair of DEST tiles: DEST gate_tile <- act(gate, up), up at gate_tile + 1.
//
// The host (common/moe_block_config.hpp) gives a kernel:
//   * the activation's kind as the named compile-time arg "expert_activation" (expert_activation_code: the Kind
//     values below);
//   * its constants as the defines EXPERT_ACTIVATION_BETA / _LINEAR_BETA (SiTU), EXPERT_ACTIVATION_LIMIT (clamped SiLU,
//     SwiGLU-OAI) and EXPERT_ACTIVATION_ALPHA (SwiGLU-OAI) - expert_activation_defines.
//
// The activations:
//   Silu         silu(gate) x up
//   SwigluOai    (clamp(up, +-limit) + 1) x g sigmoid(alpha g), g = min(gate, limit) (GPT-OSS, MiniMax M3)
//   Situ         beta tanh(gate / beta) sigmoid(gate) x linear_beta tanh(up / linear_beta) (Kimi K3)
//   ClampedSilu  silu(min(gate, limit)) x clamp(up, +-limit) (DeepSeek V4, GLM-5.3)
//
// The SFPU functions come from the LLK's SFPU headers directly: the public compute API's situ_glu_tile /
// clamped_silu_glu_tile (api/compute/situ_glu.h, clamped_silu_glu.h) run on the MATH thread with their constants fixed
// at Kimi K3's / DeepSeek V4's, and there is no public SwiGLU-OAI; the experts run the activation on the PACK thread
// (the MATH thread is busy with the next matmul) with the model's constants.
#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_PACK
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "ckernel_sfpu_situ_glu.h"
#include "ckernel_sfpu_clamped_silu_glu.h"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/swiglu_sfpu.h"
namespace ckernel::sfpu {
// The model's activation constants (the host's EXPERT_ACTIVATION_* defines), in the fields the SFPU functions read
struct ExpertActivationConstants {
    static constexpr float beta_gate = EXPERT_ACTIVATION_BETA, beta_up = EXPERT_ACTIVATION_LINEAR_BETA;
    static constexpr float limit = EXPERT_ACTIVATION_LIMIT, clamp_limit = EXPERT_ACTIVATION_LIMIT,
                           alpha = EXPERT_ACTIVATION_ALPHA;
};
}  // namespace ckernel::sfpu
#endif

namespace expert_activation {

// The activation's kind: the named compile-time arg "expert_activation" (host: expert_activation_code)
enum class Kind : uint32_t {
    Silu = 0,
    SwigluOai = 1,
    Situ = 2,
    ClampedSilu = 3,
};

// The activation's SFPU init (on the PACK thread)
template <Kind kind>
ALWI void init() {
    if constexpr (kind == Kind::Situ) {
        PACK((SFPU_BINARY_INIT_FN_NO_ARGS(situ_glu, sfpu::situ_glu_init)));
    } else if constexpr (kind == Kind::ClampedSilu) {
        PACK((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::clamped_silu_glu_init)));
    } else if constexpr (kind == Kind::SwigluOai) {
        PACK((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::swiglu_init)));
    } else {
        silu_tile_init_pack();
        PACK((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::MUL))));
    }
}

// The activation of one (gate, up) DEST tile pair, in place: DEST gate_tile <- act(gate, up), up at gate_tile + 1
template <Kind kind>
ALWI void tile_pair(uint32_t gate_tile) {
    if constexpr (kind == Kind::Situ) {
        PACK((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_situ_glu,
            (DST_ACCUM_MODE, 8, sfpu::ExpertActivationConstants),
            gate_tile,
            gate_tile + 1,
            gate_tile,
            VectorMode::RC)));
    } else if constexpr (kind == Kind::ClampedSilu) {
        PACK((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_clamped_silu_glu,
            (DST_ACCUM_MODE, 8, sfpu::ExpertActivationConstants),
            gate_tile,
            gate_tile + 1,
            gate_tile,
            VectorMode::RC)));
    } else if constexpr (kind == Kind::SwigluOai) {
        PACK((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_swiglu,
            (DST_ACCUM_MODE, 8, sfpu::ExpertActivationConstants, true),
            gate_tile,
            gate_tile + 1,
            gate_tile,
            VectorMode::RC)));
    } else {
        silu_tile_pack(gate_tile);
        PACK((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary_mul,
            (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE),
            gate_tile,
            gate_tile + 1,
            gate_tile,
            VectorMode::RC)));
    }
}

}  // namespace expert_activation
