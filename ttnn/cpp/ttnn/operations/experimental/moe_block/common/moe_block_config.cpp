// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_block_config.hpp"

#include <bit>
#include <cmath>

#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::moe_block {

namespace config_detail {
constexpr uint32_t TILE = 32;
}  // namespace config_detail

void ExpertActivation::check() const {
    if (kind == ExpertActivationKind::ClampedSilu || kind == ExpertActivationKind::SwigluOai) {
        TT_FATAL(limit.has_value() && *limit > 0, "moe_block: this expert activation needs a positive limit");
    }
    if (kind == ExpertActivationKind::SwigluOai) {
        TT_FATAL(alpha.has_value(), "moe_block: swiglu_oai needs alpha");
    }
    if (kind == ExpertActivationKind::Situ) {
        TT_FATAL(beta.has_value() && linear_beta.has_value(), "moe_block: situ needs beta and linear_beta");
    }
}

void MoEModelDims::check() const {
    const uint32_t E = num_experts, K = top_k;
    TT_FATAL(K > 0 && K <= E, "moe_block: top_k {} must be in (0, {}]", K, E);
    TT_FATAL(
        router.expert_groups > 0 && E % router.expert_groups == 0,
        "moe_block: {} experts don't split into {} groups",
        E,
        router.expert_groups);
    TT_FATAL(router.topk_groups >= 1 && router.topk_groups <= router.expert_groups, "moe_block: topk_groups");
    TT_FATAL(
        K <= router.topk_groups * (E / router.expert_groups),
        "moe_block: top_k can't exceed the experts of the kept groups");
    TT_FATAL(emb_dim % config_detail::TILE == 0, "moe_block: emb_dim {} is not a whole number of tiles", emb_dim);
    TT_FATAL(
        routed_expert_hidden_dim % config_detail::TILE == 0,
        "moe_block: routed_expert_hidden_dim {} is not a whole number of tiles",
        routed_expert_hidden_dim);
    TT_FATAL(
        shared_expert_hidden_dim.has_value() == shared_expert_activation.has_value(),
        "moe_block: a shared expert needs both its hidden dim and its activation");
    if (shared_expert_hidden_dim.has_value()) {
        TT_FATAL(
            *shared_expert_hidden_dim % config_detail::TILE == 0,
            "moe_block: shared_expert_hidden_dim not tile aligned");
        shared_expert_activation->check();
    }
    if (latent_dim.has_value()) {
        TT_FATAL(
            *latent_dim % config_detail::TILE == 0 && *latent_dim < emb_dim, "moe_block: latent_dim {}", *latent_dim);
    }
    routed_expert_activation.check();
}

uint32_t snap_bits(std::optional<float> q) {
    if (!q.has_value() || *q == 0.0f) {
        return 0;
    }
    int e = 0;
    const float m = std::frexp(*q, &e);  // q = m 2^e, m in [0.5, 1)
    TT_FATAL(m == 0.5f, "moe_block: combine_snap {} is not a power of two", *q);
    // C = 1.5 x 2^23 q = 1.5 x 2^(e - 1 + 23)
    return std::bit_cast<uint32_t>(std::ldexp(1.5f, e - 1 + 23));
}

uint32_t MoEBlockConfig::snap_bits() const { return moe_block::snap_bits(combine_snap); }

void MoEBlockConfig::check() const {
    TT_FATAL(num_links >= 1 && num_links <= 2, "moe_block: num_links {} (1 or 2)", num_links);
    TT_FATAL(
        plan_tokens() % (splits() * loops()) == 0,
        "moe_block: chunk {} does not split into {} splits x {} token loops",
        plan_tokens(),
        splits(),
        loops());
    (void)snap_bits();
}

}  // namespace ttnn::operations::experimental::moe_block
