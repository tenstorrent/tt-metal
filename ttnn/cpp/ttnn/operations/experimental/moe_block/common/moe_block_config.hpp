// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE block's typed configuration: the model's dims, the dtypes and the block's options, each with today's
// production default; ops never read the environment. (The fabric's torus tie rule is not a switch: main splits ties by
// the source's parity; mesh_routing.hpp routes by it.)

#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include <tt-metalium/base_types.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::moe_block {

enum class ExpertActivationKind : uint8_t { Silu, ClampedSilu, Situ, SwigluOai };
enum class RouterScoreFunc : uint8_t { Sigmoid, Softmax, SqrtSoftplus };

struct ExpertActivation {
    ExpertActivationKind kind = ExpertActivationKind::Silu;
    std::optional<float> limit;        // clamp to +-limit (clamped_silu, swiglu_oai)
    std::optional<float> alpha;        // swiglu_oai
    std::optional<float> beta;         // situ
    std::optional<float> linear_beta;  // situ

    void validate() const;
    bool operator==(const ExpertActivation&) const = default;
};

struct RouterSpec {
    RouterScoreFunc score_func = RouterScoreFunc::Sigmoid;
    float route_scale = 1.0f;  // routed_scaling_factor
    bool normalize_topk = true;
    uint32_t expert_groups = 1;  // group-limited routing; 1: plain top-k
    uint32_t topk_groups = 1;
    uint32_t fold = 1;  // proxy: candidates num_experts x fold, candidate c is expert c / fold (the fused head only)

    bool operator==(const RouterSpec&) const = default;
};

// What the checkpoint says (MoEModel). K3Dims (H, latent, I, E, K, act) is the subset the exchange carries:
// H = emb_dim, latent = routed_width(), I = routed_expert_hidden_dim, E = num_experts, K = top_k.
struct MoEModelDims {
    uint32_t num_experts = 0;               // E
    uint32_t top_k = 0;                     // K
    uint32_t emb_dim = 0;                   // H (residual width)
    uint32_t routed_expert_hidden_dim = 0;  // I
    ExpertActivation routed_expert_activation;
    RouterSpec router;
    std::optional<uint32_t> shared_expert_hidden_dim;  // nullopt: no shared expert
    std::optional<ExpertActivation> shared_expert_activation;
    std::optional<uint32_t> latent_dim;  // latent MoE (K3): the routed experts run at this width
    bool latent_norm = true;
    float norm_eps = 1e-5f;

    // Hr: the width on the wire and in the experts.
    uint32_t routed_width() const { return latent_dim.value_or(emb_dim); }
    void validate() const;
    bool operator==(const MoEModelDims&) const = default;
};

struct MoEDtypes {
    tt::tt_metal::DataType routed_weights = tt::tt_metal::DataType::BFLOAT4_B;
    tt::tt_metal::DataType shared_weights = tt::tt_metal::DataType::BFLOAT8_B;
    tt::tt_metal::DataType shared_activations = tt::tt_metal::DataType::BFLOAT16;
    tt::tt_metal::MathFidelity shared_fidelity = tt::tt_metal::MathFidelity::HiFi2;
    tt::tt_metal::DataType projection_weights = tt::tt_metal::DataType::BFLOAT8_B;  // latent down / up
    // the fused tail's up projection; never LoFi
    tt::tt_metal::MathFidelity tail_fidelity = tt::tt_metal::MathFidelity::HiFi2;
    tt::tt_metal::DataType router_weights = tt::tt_metal::DataType::BFLOAT16;

    bool operator==(const MoEDtypes&) const = default;
};

// The block's options. Geometry (mesh shape, torus) is derived from the mesh and the fabric config by
// MoEBlockResources, never passed in.
struct MoEBlockConfig {
    bool deterministic = true;          // the router's fin in a fixed order (RT_ARR_ORDERED)
    std::optional<float> combine_snap;  // power of two q: every routed term rounded to a multiple of q
    bool head_dram_fallback = true;     // the head planner may take its DRAM fallback

    // Geometry inputs that are not derived
    uint32_t num_links = 2;     // links per direction, the same on every axis
    uint32_t chunk_tokens = 0;  // the global chunk, all chips together

    // Where the persistent words live
    bool persist_l1_small = false;  // in L1_SMALL (else L1)
    // With persist_l1_small: the persistent words (MoEBlockResources::persistent_words) packed into one L1_SMALL slab
    // of at most this many bytes a core, word sets on disjoint cores sharing bytes (0: one L1_SMALL tensor a word set,
    // each reserving its bytes on every core). A set that no longer fits gets its own tensor (logged). 2304: every word
    // set of the block at 8x4 chunk 5120 fits, the fused tail's included (Kimi-K3: 1376 B used where the tail's 912 B
    // go; GLM-5.3: 2048 B) - a mesh opens with L1_SMALL for this slab and the model's own use (moe_impl.py)
    uint32_t words_slab_bytes = 2304;

    // The fused head's hand-off stride: 4 B in L1_SMALL, else 16 B.
    uint32_t handoff_bytes() const { return persist_l1_small ? 4u : 16u; }
    // The combine's snap constant: the fp32 bits of C = 1.5 x 2^23 q, 0 when combine_snap is unset.
    uint32_t snap_bits() const;
    void validate() const;
    bool operator==(const MoEBlockConfig&) const = default;
};

uint32_t snap_bits(std::optional<float> q);

}  // namespace ttnn::operations::experimental::moe_block
