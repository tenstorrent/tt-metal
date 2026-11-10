// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE block's typed configuration: the model's dims, the dtypes and the block's options, each with today's
// production default; ops never read the environment. (How the fabric routes - its dimension order, its ring ties - is
// not a switch: the ops ask the fabric, mesh_routing.hpp FirstHops.)

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/base_types.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::moe_block {

enum class ExpertActivationKind : uint8_t { Silu, ClampedSilu, Situ, SwigluOai };
enum class RouterScoreFunc : uint8_t { Sigmoid, SqrtSoftplus };

struct ExpertActivation {
    ExpertActivationKind kind = ExpertActivationKind::Silu;
    std::optional<float> limit;        // clamp to +-limit (clamped_silu, swiglu_oai)
    std::optional<float> alpha;        // swiglu_oai
    std::optional<float> beta;         // situ
    std::optional<float> linear_beta;  // situ

    void check() const;  // TT_FATAL on an invalid field
    bool operator==(const ExpertActivation&) const = default;
};

struct RouterSpec {
    RouterScoreFunc score_func = RouterScoreFunc::Sigmoid;
    float route_scale = 1.0f;  // routed_scaling_factor
    uint32_t fold = 1;  // proxy: candidates num_experts x fold, candidate c is expert c / fold (the MoE head only)

    void check() const;  // TT_FATAL on an invalid field
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
    void check() const;  // TT_FATAL on an invalid field
    bool operator==(const MoEModelDims&) const = default;
};

struct MoEDtypes {
    tt::tt_metal::DataType routed_weights = tt::tt_metal::DataType::BFLOAT4_B;
    tt::tt_metal::DataType shared_weights = tt::tt_metal::DataType::BFLOAT8_B;
    tt::tt_metal::MathFidelity shared_fidelity = tt::tt_metal::MathFidelity::HiFi2;

    bool operator==(const MoEDtypes&) const = default;
};

// The block's options. Geometry (mesh shape, torus) is derived from the mesh and the fabric config by
// MoEBlockResources, never passed in.
struct MoEBlockConfig {
    // power of two q: every routed term rounded to a multiple of q, so the combine's fp32 sums are bit-identical in
    // any landing order (costs no measurable time); 0: off (its one spelling) - long chunks are then nondeterministic
    float combine_snap = 0x1p-14f;
    // the MoE head's router iterations (HeadLayout.router_iterations): 0 the planner's (by cost, each layout with the
    // fewest that fit), n at least n (tests: the iterated router where the planner would not pick it)
    uint32_t head_router_iterations = 0;

    // Geometry inputs that are not derived
    // links per direction, the same on every axis: 1 or 2. The head, the dispatch, the combine and the tail's
    // all-to-all take either; the tail's engine (a latent model's: Kimi-K3) needs 2 (checked at MoEBlockResources
    // creation).
    uint32_t num_links = 2;
    // The mesh axis SP (the token shards) runs on; TP on the other. 1: the 4x4 tray-half stage (SP on the axis-1 ring,
    // TP on the axis-0 line) - the ops then work on the transposed mesh (mesh_routing.hpp MeshTopology.transposed).
    uint32_t sp_axis = 0;
    uint32_t chunk_tokens = 0;  // the global chunk, all chips together (must be set: > 0)
    // The block's splits (MoEBlock's choice, MoEBlockResources::set_chunk_splits): the whole block (head, dispatch,
    // experts + combine, tail) runs a chunk as chunk_splits pieces of chunk_tokens / chunk_splits tokens - each piece
    // reads every expert's weights again, so only where the chunk does not fit in one call (MoEBlock warns). The ops
    // plan for one piece (split_tokens); a call names its piece (split) for a padded chunk's real rows. 0 is 1.
    uint32_t chunk_splits = 0;
    // Test only: the MoE head's row blocks (its kernels' loop over the chunk's rows, a block's rows each). 0 (always,
    // outside tests): the head's planner picks the count from its L1 fit - 1 wherever the chunk fits at once.
    uint32_t force_head_row_blocks = 0;
    // Padding rows: the ops run the chunk with pad_tokens more tokens, pad_tokens / SP rows appended to each SP row's
    // (a chunk whose rows a mesh row are not a multiple of 32 x TP - the head's rows rule); they are padding tokens of
    // a padded chunk (not routed), the MoE head always takes the chunk's actual_start / actual_end (MoEBlock gives
    // (0, chunk) for an unpadded one) and MoEBlock drops their rows of the output. 0: MoEBlockResources sets the
    // padding the chunk needs (none for whole rows).
    uint32_t pad_tokens = 0;
    // The chunks packed in a forward (common/prefill/prefill_chunk.py: forward_tokens_budget / chunk_size):
    // chunk_tokens is the forward's tokens; each SP row holds every chunk's chunk_tokens / (forward_chunks x SP) rows
    // back to back, each chunk in main's rotated layout with its own actual_start / actual_end (the forward's metadata
    // words).
    uint32_t forward_chunks = 1;
    uint32_t splits() const { return chunk_splits == 0 ? 1u : chunk_splits; }
    // the tokens the ops run: the chunk and its padding rows
    uint32_t plan_tokens() const { return chunk_tokens + pad_tokens; }
    // the tokens every op is planned for: one split's
    uint32_t split_tokens() const { return plan_tokens() / splits(); }

    // Where the persistent words live
    bool persist_l1_small = false;  // in L1_SMALL (else L1)
    // With persist_l1_small: the persistent words (MoEBlockResources::persistent_words) packed into one L1_SMALL slab
    // of at most this many bytes a core, word sets on disjoint cores sharing bytes (0: one L1_SMALL tensor a word set,
    // each reserving its bytes on every core). A set that no longer fits gets its own tensor (logged). 2304: every word
    // set of the block at 8x4 chunk 5120 fits, the MoE tail's included (Kimi-K3: 1376 B used where the tail's 912 B
    // go; GLM-5.3: 2048 B) - a mesh opens with L1_SMALL for this slab and the model's own use (moe_impl.py)
    uint32_t words_slab_bytes = 2304;

    // The MoE head's hand-off stride: 4 B in L1_SMALL, else 16 B.
    uint32_t handoff_bytes() const { return persist_l1_small ? 4u : 16u; }
    // The combine's snap constant: the fp32 bits of C = 1.5 x 2^23 q, 0 when combine_snap is 0 (off).
    uint32_t snap_bits() const;
    void check() const;  // TT_FATAL on an invalid field
    bool operator==(const MoEBlockConfig&) const = default;
};

uint32_t snap_bits(float q);  // q: 0 (off) or a positive power of two

// The experts' GLU activation as the device kernels take it (common/kernels/expert_activation.hpp, the shared and the
// routed experts alike): its kind as the named compile-time arg kExpertActivationArgName (expert_activation_code: 0
// SiLU, 1 SwiGLU-OAI, 2 SiTU, 3 clamped SiLU) and its constants as the EXPERT_ACTIVATION_* defines
// (expert_activation_defines; a constant the model leaves unset: default_expert_activation's).
inline constexpr const char* kExpertActivationArgName = "expert_activation";
uint32_t expert_activation_code(ExpertActivationKind kind);
std::vector<std::pair<std::string, std::string>> expert_activation_defines(const ExpertActivation& activation);
// The activation of a code with its default constants (SwiGLU-OAI limit 7, alpha 1.702; SiTU betas 4 and 25; clamped
// SiLU limit 10).
ExpertActivation default_expert_activation(uint32_t code);

// The routed experts' activation code (expert_activation_code). The flat expert takes only the code and builds with
// the default constants: TT_FATAL on a model whose constants differ, instead of computing another activation.
uint32_t routed_activation_code(const ExpertActivation& act);

}  // namespace ttnn::operations::experimental::moe_block
