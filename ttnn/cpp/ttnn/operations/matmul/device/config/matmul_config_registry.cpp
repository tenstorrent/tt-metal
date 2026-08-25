// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/matmul_config_registry.hpp"

#include <atomic>
#include <bit>
#include <utility>

#include "ttnn/operation.hpp"

namespace ttnn::operations::matmul::registry {
namespace {

constexpr std::uint8_t kModeUninitialized = 0xff;
std::atomic<std::uint8_t> frozen_mode{kModeUninitialized};

struct AtomicDomainStats {
    std::atomic<std::uint64_t> resolution_attempts{0};
    std::atomic<std::uint64_t> certified_hits{0};
    std::atomic<std::uint64_t> shadow_would_hits{0};
    std::atomic<std::uint64_t> applied_hits{0};
    std::atomic<std::uint64_t> fallbacks{0};
    std::array<std::atomic<std::uint64_t>, kResolutionReasonCount> reasons{};
};

std::array<AtomicDomainStats, kOperationDomainCount> stats;

constexpr std::size_t index(const OperationDomain domain) { return static_cast<std::size_t>(domain); }
constexpr std::size_t index(const ResolutionReason reason) { return static_cast<std::size_t>(reason); }

const Recipe* lookup_exact(const MatmulRegistryRequest& request) noexcept {
    // B0/B2 deliberately carry no native or generated production entry. B1
    // replaces this stub with POD descriptor lookup plus separate fallible
    // native materialization; an allocating Recipe must never become the table.
    static_cast<void>(request);
    return nullptr;
}

std::optional<tt::tt_metal::Tile> transpose_matmul_tile(const tt::tt_metal::Tile& tile, const bool transpose) {
    if (!transpose) {
        return tile;
    }

    const bool transpose_of_faces = tile.get_transpose_of_faces();
    if (transpose_of_faces && !tile.get_transpose_within_face()) {
        return std::nullopt;
    }
    return tt::tt_metal::Tile({tile.get_width(), tile.get_height()}, !transpose_of_faces);
}

}  // namespace

Mode current_mode() noexcept {
    auto value = frozen_mode.load(std::memory_order_acquire);
    if (value != kModeUninitialized) {
        return static_cast<Mode>(value);
    }

    const auto configured = ttnn::CONFIG.get<"matmul_registry_mode">();
    const auto raw_configured_value = static_cast<std::uint8_t>(configured);
    const auto configured_value = raw_configured_value <= static_cast<std::uint8_t>(Mode::On)
                                      ? raw_configured_value
                                      : static_cast<std::uint8_t>(Mode::Off);
    if (frozen_mode.compare_exchange_strong(
            value, configured_value, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return static_cast<Mode>(configured_value);
    }
    return static_cast<Mode>(value);
}

void reset_startup_mode_for_testing() noexcept { frozen_mode.store(kModeUninitialized, std::memory_order_release); }

CallSemantics addmm_call_semantics(const float alpha, const float beta) noexcept {
    return CallSemantics{
        .domain = OperationDomain::Addmm,
        .alpha_f32_bits = std::bit_cast<std::uint32_t>(alpha),
        .beta_f32_bits = std::bit_cast<std::uint32_t>(beta)};
}

ResolvedMatmulIoContract resolve_matmul_io_contract(const IoContractRequest& request) {
    auto output_memory_config = request.requested_output_memory_config;
    auto output_dtype = request.requested_output_dtype.value_or(request.input_a_dtype);

    if (request.optional_output.has_value()) {
        const auto& optional_output = request.optional_output.value();
        if (output_memory_config == tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
            output_memory_config = optional_output.memory_config;
        } else if (output_memory_config != optional_output.memory_config) {
            return {
                .status = IoContractStatus::OptionalOutputMemoryMismatch,
                .output_memory_config = output_memory_config,
                .output_dtype = output_dtype,
                .output_tile = request.input_a_tile,
                .uses_optional_output = true};
        }

        if (request.requested_output_dtype.has_value() &&
            request.requested_output_dtype.value() != optional_output.dtype) {
            return {
                .status = IoContractStatus::OptionalOutputDtypeMismatch,
                .output_memory_config = output_memory_config,
                .output_dtype = output_dtype,
                .output_tile = request.input_a_tile,
                .uses_optional_output = true};
        }
        output_dtype = optional_output.dtype;
    }

    const auto input_a_tile = transpose_matmul_tile(request.input_a_tile, request.transpose_a);
    const auto input_b_tile = transpose_matmul_tile(request.input_b_tile, request.transpose_b);
    if (!input_a_tile.has_value() || !input_b_tile.has_value()) {
        return {
            .status = IoContractStatus::InvalidTransposeTile,
            .output_memory_config = output_memory_config,
            .output_dtype = output_dtype,
            .output_tile = request.input_a_tile,
            .uses_optional_output = request.optional_output.has_value()};
    }

    if (request.requested_output_tile.has_value() && request.optional_output.has_value()) {
        return {
            .status = IoContractStatus::OutputTileConflict,
            .output_memory_config = output_memory_config,
            .output_dtype = output_dtype,
            .output_tile = request.requested_output_tile.value(),
            .uses_optional_output = true};
    }

    const auto output_tile = request.requested_output_tile.has_value() ? request.requested_output_tile.value()
                             : request.optional_output.has_value()
                                 ? request.optional_output->tile
                                 : tt::tt_metal::Tile({input_a_tile->get_height(), input_b_tile->get_width()});
    return {
        .status = IoContractStatus::Resolved,
        .output_memory_config = output_memory_config,
        .output_dtype = output_dtype,
        .output_tile = output_tile,
        .uses_optional_output = request.optional_output.has_value()};
}

ExecutionAction execution_action(const Mode mode, const Resolution& resolution) noexcept {
    if (resolution.reason != ResolutionReason::CertifiedMatch || resolution.recipe == nullptr ||
        !has_consistent_untilize_out(*resolution.recipe)) {
        return ExecutionAction::Fallback;
    }
    if (mode == Mode::On) {
        return ExecutionAction::ApplyRecipe;
    }
    return mode == Mode::Shadow ? ExecutionAction::ObserveOnly : ExecutionAction::Fallback;
}

std::optional<ttnn::prim::MatmulParams> materialize_parameters_for_execution(
    const Mode mode, const Resolution& resolution, const ttnn::prim::MatmulParams& legacy_parameters) {
    if (execution_action(mode, resolution) != ExecutionAction::ApplyRecipe) {
        return std::nullopt;
    }

    auto materialized = legacy_parameters;
    materialized.program_config = resolution.recipe->program_config;
    materialized.compute_kernel_config = resolution.recipe->compute_kernel_config;
    materialized.untilize_out = resolution.recipe->untilize_out;
    return materialized;
}

bool has_consistent_untilize_out(const Recipe& recipe) noexcept {
    if (const auto* config = std::get_if<MatmulMultiCoreReuseMultiCast1DProgramConfig>(&recipe.program_config)) {
        return config->untilize_out == recipe.untilize_out;
    }
    return !recipe.untilize_out;
}

static Resolution resolve_impl(
    const Mode mode,
    const MatmulRegistryRequest& request,
    const Eligibility& eligibility,
    const MatmulRegistryRequest* synthetic_request,
    const Recipe* synthetic_recipe) noexcept {
    if (mode == Mode::Off) {
        return {.reason = ResolutionReason::Disabled};
    }
    if (request.schema_version != 1) {
        return {.reason = ResolutionReason::IncompleteRequest};
    }
    if (request.call != eligibility.call || request.transpose_a != eligibility.transpose_a ||
        request.transpose_b != eligibility.transpose_b || request.has_bias != eligibility.has_bias ||
        request.has_activation != eligibility.has_activation || request.untilize_out != eligibility.untilize_out ||
        request.bcast_batch.has_value() != eligibility.has_bcast_batch ||
        request.run_batched != eligibility.input_b_batched ||
        request.has_activation != request.activation_op.has_value()) {
        return {.reason = ResolutionReason::InconsistentRequest};
    }
    if (eligibility.call.domain == OperationDomain::IneligibleSharedCaller) {
        return {.reason = ResolutionReason::IneligibleOperationDomain};
    }
    const bool is_addmm = eligibility.call.domain == OperationDomain::Addmm;
    if (is_addmm != (eligibility.call.alpha_f32_bits.has_value() && eligibility.call.beta_f32_bits.has_value())) {
        return {.reason = ResolutionReason::MalformedOperationSemantics};
    }
    if (eligibility.io_contract_status != IoContractStatus::Resolved) {
        return {.reason = ResolutionReason::InconsistentIoContract};
    }
    if (eligibility.has_program_config || eligibility.has_compute_kernel_config || eligibility.has_user_core_grid) {
        return {.reason = ResolutionReason::ExplicitOverride};
    }
    const bool unsupported_bias = eligibility.has_bias && eligibility.call.domain != OperationDomain::Linear;
    const bool unsupported_activation =
        eligibility.has_activation && eligibility.call.domain != OperationDomain::Linear;
    const bool unsupported_transpose =
        (eligibility.transpose_a || eligibility.transpose_b) && eligibility.call.domain != OperationDomain::Linear;
    if (unsupported_bias || unsupported_activation || eligibility.has_optional_output || eligibility.has_output_tile ||
        eligibility.has_global_cb || eligibility.has_sub_device || eligibility.has_bcast_batch ||
        eligibility.untilize_out || eligibility.input_a_sharded || eligibility.input_b_sharded ||
        eligibility.output_sharded || eligibility.input_b_batched || unsupported_transpose) {
        return {.reason = ResolutionReason::UnsupportedSemantics};
    }

    // V1's first generated table is dense-only. Linear and addmm remain explicit
    // empty domains until their bias/activation/output semantics have their own
    // certified entries; they can never fall through into dense recipes.
    if (request.call.domain != OperationDomain::DenseMatmul) {
        return {.reason = ResolutionReason::EmptyRegistry};
    }

    if (synthetic_request != nullptr && synthetic_recipe != nullptr && *synthetic_request == request) {
        return {.reason = ResolutionReason::CertifiedMatch, .recipe = synthetic_recipe};
    }
    if (const auto* recipe = lookup_exact(request); recipe != nullptr) {
        return {.reason = ResolutionReason::CertifiedMatch, .recipe = recipe};
    }

    // The generated certified table intentionally starts empty. Shadow and On
    // therefore both preserve the existing selector and cache identity today.
    return {.reason = ResolutionReason::EmptyRegistry};
}

Resolution resolve(const Mode mode, const MatmulRegistryRequest& request, const Eligibility& eligibility) noexcept {
    return resolve_impl(mode, request, eligibility, nullptr, nullptr);
}

DispatchResult resolve_for_dispatch(
    const Mode mode,
    const std::optional<MatmulRegistryRequest>& request,
    const Eligibility& eligibility,
    const ttnn::prim::MatmulParams& legacy_parameters,
    const ResolverFunction resolver) {
    auto resolution = Resolution{.reason = ResolutionReason::Disabled};
    if (mode != Mode::Off) {
        resolution = request.has_value() && resolver != nullptr
                         ? resolver(mode, request.value(), eligibility)
                         : Resolution{.reason = ResolutionReason::IncompleteRequest};
    }

    const auto action = execution_action(mode, resolution);
    auto materialized_parameters = materialize_parameters_for_execution(mode, resolution, legacy_parameters);
    return DispatchResult{
        .resolution = resolution, .action = action, .materialized_parameters = std::move(materialized_parameters)};
}

Resolution resolve_with_synthetic_candidate_for_testing(
    const Mode mode,
    const MatmulRegistryRequest& request,
    const Eligibility& eligibility,
    const MatmulRegistryRequest& candidate_request,
    const Recipe& candidate_recipe) noexcept {
    return resolve_impl(mode, request, eligibility, &candidate_request, &candidate_recipe);
}

void record_resolution(
    const Mode mode,
    const OperationDomain domain,
    const Resolution& resolution,
    const ExecutionAction action) noexcept {
    if (mode == Mode::Off || index(domain) >= stats.size() || index(resolution.reason) >= kResolutionReasonCount) {
        return;
    }

    auto& domain_stats = stats[index(domain)];
    domain_stats.resolution_attempts.fetch_add(1, std::memory_order_relaxed);
    domain_stats.reasons[index(resolution.reason)].fetch_add(1, std::memory_order_relaxed);
    if (resolution.reason == ResolutionReason::CertifiedMatch && resolution.recipe != nullptr) {
        domain_stats.certified_hits.fetch_add(1, std::memory_order_relaxed);
    }
    switch (action) {
        case ExecutionAction::ObserveOnly:
            domain_stats.shadow_would_hits.fetch_add(1, std::memory_order_relaxed);
            break;
        case ExecutionAction::ApplyRecipe: domain_stats.applied_hits.fetch_add(1, std::memory_order_relaxed); break;
        case ExecutionAction::Fallback: domain_stats.fallbacks.fetch_add(1, std::memory_order_relaxed); break;
    }
}

StatsSnapshot stats_snapshot() noexcept {
    StatsSnapshot snapshot;
    const auto mode = frozen_mode.load(std::memory_order_acquire);
    snapshot.mode_is_frozen = mode != kModeUninitialized;
    snapshot.frozen_mode = snapshot.mode_is_frozen ? static_cast<Mode>(mode) : Mode::Off;

    for (std::size_t domain = 0; domain < stats.size(); ++domain) {
        const auto& source = stats[domain];
        auto& destination = snapshot.domains[domain];
        destination.resolution_attempts = source.resolution_attempts.load(std::memory_order_relaxed);
        destination.certified_hits = source.certified_hits.load(std::memory_order_relaxed);
        destination.shadow_would_hits = source.shadow_would_hits.load(std::memory_order_relaxed);
        destination.applied_hits = source.applied_hits.load(std::memory_order_relaxed);
        destination.fallbacks = source.fallbacks.load(std::memory_order_relaxed);
        for (std::size_t reason = 0; reason < kResolutionReasonCount; ++reason) {
            destination.reasons[reason] = source.reasons[reason].load(std::memory_order_relaxed);
        }
    }
    return snapshot;
}

void reset_stats_for_testing() noexcept {
    for (auto& domain : stats) {
        domain.resolution_attempts.store(0, std::memory_order_relaxed);
        domain.certified_hits.store(0, std::memory_order_relaxed);
        domain.shadow_would_hits.store(0, std::memory_order_relaxed);
        domain.applied_hits.store(0, std::memory_order_relaxed);
        domain.fallbacks.store(0, std::memory_order_relaxed);
        for (auto& reason : domain.reasons) {
            reason.store(0, std::memory_order_relaxed);
        }
    }
}

}  // namespace ttnn::operations::matmul::registry
