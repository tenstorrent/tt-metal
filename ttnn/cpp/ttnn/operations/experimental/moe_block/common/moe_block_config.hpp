// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE block's typed configuration. Every switch the Python block read from the environment (SPC_*, PERSIST_L1) is
// a field here with today's production default; ops never read the environment. (The fabric's torus tie rule is not
// a switch: main splits ties by the source's parity, the reference's default; mesh_routing.hpp routes by it.)
// Reference: models/demos/moe_block/moe_block.py (MoEModel, RouterConfig, ExpertActivation, MoEDtypes, MoEOptions)
// and ttnn/ttnn/bringup/combine_rf/sparse_combine_rf.py (CombineTuning) on mstaletovic/moe-trim.

#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include <tt-metalium/base_types.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::moe_block {

// Where a fabric port core sits relative to its link's Ethernet core (SPC_PLACE).
enum class PortPlacement : uint8_t {
    Noc1 = 0,      // fewest NoC1 hops to the Ethernet core, per chip (default: the sender is on NoC1)
    Noc0 = 1,      // fewest NoC0 hops (for snoc = 0: the sender on NoC0)
    Auto = 2,      // Noc1 if it leaves >= 8 allowed cells free, else Uniform0
    Uniform0 = 3,  // one logical cell per (direction, link) on every chip: fewest NoC0 hops summed over the chips
    Uniform1 = 4,  // the same on NoC1
};

enum class ExpertActivationKind : uint8_t { Silu, ClampedSilu, Situ, SwigluOai };
enum class RouterScoreFunc : uint8_t { Sigmoid, Softmax, SqrtSoftplus };
enum class ResidualLayout : uint8_t {
    SpTp,    // main's layout [S/SP, H/TP] a chip
    Owners,  // each chip's own tokens at full width [S/N, H]
};

// The combine's tuning switches (sparse_combine_rf.CombineTuning, SPC_* names), today's defaults. All structural
// (compile-time args / RISC assignment / placement), so all hashed.
struct CombineTuning {
    uint32_t snoc = 1;      // SPC_SNOC: 1 the sender on NoC1 (RISCV_0); 0 the sender on NoC0 (RISCV_1)
    uint32_t la = 0;        // SPC_LA: look-ahead past a row whose ring is full (0 / 1: strict queue order; max 64)
    uint32_t far = 0;       // SPC_FAR: within the look-ahead window, rows going >= far hops first (only with la > 1)
    bool credauto = false;  // SPC_CREDAUTO: credits only at the end when every pool holds its call
    uint32_t credthr = 8;   // SPC_CREDTHR: credit return threshold
    uint32_t credidle = 0;  // SPC_CREDIDLE: idle credit sends wait for this many rows (0: the combine's choice)
    uint32_t credage = 0;   // SPC_CREDAGE: ... or for the oldest to be this many cycles old (used with credidle)
    bool gbgate = false;    // SPC_GBGATE: busy-pass credit scan only when the total pending reaches credthr
    PortPlacement place = PortPlacement::Noc1;  // SPC_PLACE

    bool operator==(const CombineTuning&) const = default;
};

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

// The block's options (MoEOptions) + the machine-side switches. Geometry (mesh shape, torus) is derived from the mesh
// and the fabric config by MoEBlockResources, never passed in.
struct MoEBlockConfig {
    // MoEOptions
    bool experts_during_dispatch = false;  // the dispatch as the combine ports' first phase (in the experts' program)
    bool deterministic = true;             // router fin in a fixed order (RouterSpec.ordered / RT_ARR_ORDERED)
    std::optional<float> combine_snap;     // power of two q: every routed term rounded to a multiple of q
    bool head_dram_fallback = true;        // head_plan fit(fallback=)
    bool router_split = false;             // RT_SPLIT / SE_RTR_SPLIT
    ResidualLayout residual = ResidualLayout::SpTp;

    // Geometry inputs that are not derived
    uint32_t num_links = 2;     // links per direction, the same on every axis
    uint32_t chunk_tokens = 0;  // the global chunk, all chips together
    bool zigzag = false;        // the residual's sequence order

    // Machine-side switches (env reads in the reference)
    bool persist_l1_small = false;  // PERSIST_L1=small: persistent words + packed semaphores in L1_SMALL
    // With persist_l1_small: the persistent words (MoEBlockResources::persistent_words) packed into one L1_SMALL slab of
    // at most this many bytes a core, word sets on disjoint cores sharing bytes (0: one L1_SMALL tensor a word set,
    // each reserving its bytes on every core). A set that no longer fits gets its own tensor (logged).
    uint32_t words_slab_bytes = 2048;
    CombineTuning combine_tuning;

    // PERSIST_L1 hand-off stride (persist_l1.handoff_bytes): 4 B in L1_SMALL, else 16 B.
    uint32_t handoff_bytes() const { return persist_l1_small ? 4u : 16u; }
    // __SNAP__: the fp32 bits of C = 1.5 x 2^23 q, 0 when combine_snap is unset (sparse_combine_rf._snap_bits).
    uint32_t snap_bits() const;
    void validate() const;
    bool operator==(const MoEBlockConfig&) const = default;
};

uint32_t snap_bits(std::optional<float> q);

}  // namespace ttnn::operations::experimental::moe_block
