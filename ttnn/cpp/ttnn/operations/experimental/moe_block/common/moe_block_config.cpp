// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_block_config.hpp"

#include <bit>
#include <charconv>
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

void RouterSpec::check() const {
    TT_FATAL(fold > 0, "moe_block: router fold {} (candidates num_experts x fold) must be > 0", fold);
    TT_FATAL(
        std::isfinite(route_scale) && route_scale > 0.0f,
        "moe_block: router route_scale {} must be finite and > 0",
        route_scale);
}

void MoEModelDims::check() const {
    const uint32_t E = num_experts, K = top_k;
    TT_FATAL(K > 0 && K <= E, "moe_block: top_k {} must be in (0, {}]", K, E);
    TT_FATAL(
        emb_dim > 0 && emb_dim % config_detail::TILE == 0,
        "moe_block: emb_dim {} is not a positive whole number of tiles",
        emb_dim);
    TT_FATAL(
        routed_expert_hidden_dim > 0 && routed_expert_hidden_dim % config_detail::TILE == 0,
        "moe_block: routed_expert_hidden_dim {} is not a positive whole number of tiles",
        routed_expert_hidden_dim);
    TT_FATAL(
        shared_expert_hidden_dim.has_value() == shared_expert_activation.has_value(),
        "moe_block: a shared expert needs both its hidden dim and its activation");
    if (shared_expert_hidden_dim.has_value()) {
        TT_FATAL(
            *shared_expert_hidden_dim > 0 && *shared_expert_hidden_dim % config_detail::TILE == 0,
            "moe_block: shared_expert_hidden_dim {} is not a positive whole number of tiles",
            *shared_expert_hidden_dim);
        shared_expert_activation->check();
    }
    if (latent_dim.has_value()) {
        TT_FATAL(
            *latent_dim > 0 && *latent_dim % config_detail::TILE == 0 && *latent_dim < emb_dim,
            "moe_block: latent_dim {} (a positive whole number of tiles, < emb_dim {})",
            *latent_dim,
            emb_dim);
    }
    routed_expert_activation.check();
    router.check();
}

uint32_t snap_bits(float q) {
    if (q == 0.0f) {
        return 0;
    }
    int e = 0;
    const float m = std::frexp(q, &e);  // q = m 2^e, m in [0.5, 1) (negative, inf, NaN: m is not 0.5)
    TT_FATAL(m == 0.5f, "moe_block: combine_snap {} is not 0 (off) or a positive power of two", q);
    // C = 1.5 x 2^23 q = 1.5 x 2^(e - 1 + 23)
    return std::bit_cast<uint32_t>(std::ldexp(1.5f, e - 1 + 23));
}

uint32_t expert_activation_code(ExpertActivationKind kind) {
    switch (kind) {
        case ExpertActivationKind::Silu: return 0;
        case ExpertActivationKind::SwigluOai: return 1;
        case ExpertActivationKind::Situ: return 2;
        case ExpertActivationKind::ClampedSilu: return 3;
    }
    TT_THROW("moe_block: unknown expert activation");
}

ExpertActivation default_expert_activation(uint32_t code) {
    switch (code) {
        case 0: return ExpertActivation{.kind = ExpertActivationKind::Silu};
        case 1: return ExpertActivation{.kind = ExpertActivationKind::SwigluOai, .limit = 7.0f, .alpha = 1.702f};
        case 2: return ExpertActivation{.kind = ExpertActivationKind::Situ, .beta = 4.0f, .linear_beta = 25.0f};
        case 3: return ExpertActivation{.kind = ExpertActivationKind::ClampedSilu, .limit = 10.0f};
        default: TT_THROW("moe_block: expert activation code {}: 0 SiLU, 1 SwiGLU-OAI, 2 SiTU, 3 clamped SiLU", code);
    }
}

namespace config_detail {
// A float as a C++ literal: the shortest round-trip decimal, fixed notation, ".0" if whole, then "f".
std::string float_literal(float value) {
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::fixed);
    std::string literal(buffer, result.ptr);
    if (literal.find('.') == std::string::npos) {
        literal += ".0";
    }
    return literal + "f";
}
}  // namespace config_detail

std::vector<std::pair<std::string, std::string>> expert_activation_defines(const ExpertActivation& activation) {
    const ExpertActivation defaults = default_expert_activation(expert_activation_code(activation.kind));
    const auto value = [](const std::optional<float>& set, const std::optional<float>& fallback, float last) {
        return config_detail::float_literal(set.value_or(fallback.value_or(last)));
    };
    // (every define, whatever the kind: the kernel's constants struct names them all)
    return {
        {"EXPERT_ACTIVATION_BETA", value(activation.beta, defaults.beta, 4.0f)},
        {"EXPERT_ACTIVATION_LINEAR_BETA", value(activation.linear_beta, defaults.linear_beta, 25.0f)},
        {"EXPERT_ACTIVATION_LIMIT", value(activation.limit, defaults.limit, 10.0f)},
        {"EXPERT_ACTIVATION_ALPHA", value(activation.alpha, defaults.alpha, 1.702f)}};
}

uint32_t routed_activation_code(const ExpertActivation& act) {
    act.check();
    auto is = [](const std::optional<float>& v, float want) {
        return v.has_value() && std::fabs(*v - want) <= 1e-6f * want;
    };
    switch (act.kind) {
        case ExpertActivationKind::Silu: break;
        case ExpertActivationKind::SwigluOai:
            TT_FATAL(
                is(act.limit, 7.0f) && is(act.alpha, 1.702f),
                "moe_block: the routed experts' SwiGLU-OAI is built for limit 7, alpha 1.702 (got limit {}, alpha {})",
                act.limit.value_or(0.0f),
                act.alpha.value_or(0.0f));
            break;
        case ExpertActivationKind::Situ:
            TT_FATAL(
                is(act.beta, 4.0f) && is(act.linear_beta, 25.0f),
                "moe_block: the routed experts' SiTU is built for beta 4, linear_beta 25 (got {}, {})",
                act.beta.value_or(0.0f),
                act.linear_beta.value_or(0.0f));
            break;
        case ExpertActivationKind::ClampedSilu:
            TT_FATAL(
                is(act.limit, 10.0f),
                "moe_block: the routed experts' clamped SiLU is built for limit 10 (got {})",
                act.limit.value_or(0.0f));
            break;
    }
    return expert_activation_code(act.kind);
}

uint32_t MoEBlockConfig::snap_bits() const { return moe_block::snap_bits(combine_snap); }

void MoEBlockConfig::check() const {
    TT_FATAL(num_links >= 1 && num_links <= 2, "moe_block: num_links {} (1 or 2)", num_links);
    TT_FATAL(sp_axis <= 1, "moe_block: sp_axis {} (0 or 1)", sp_axis);
    TT_FATAL(chunk_tokens > 0, "moe_block: MoEBlockConfig.chunk_tokens is not set (0)");
    TT_FATAL(
        plan_tokens() % splits() == 0, "moe_block: chunk {} does not split into {} splits", plan_tokens(), splits());
    TT_FATAL(
        forward_chunks >= 1 && forward_chunks <= 8 && chunk_tokens % forward_chunks == 0,
        "moe_block: a forward of {} tokens as {} chunks (1 .. 8, whole chunks)",
        chunk_tokens,
        forward_chunks);
    (void)snap_bits();
}

}  // namespace ttnn::operations::experimental::moe_block
