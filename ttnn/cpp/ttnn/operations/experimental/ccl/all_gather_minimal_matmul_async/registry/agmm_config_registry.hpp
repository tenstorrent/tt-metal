// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include "agmm_registry_descriptor.hpp"
#include "ttnn/config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::experimental::all_gather_minimal_matmul_registry {

using Mode = ttnn::MatmulRegistryMode;

enum class ResolutionReason : std::uint8_t {
    Disabled,
    TraceCaptureUnsupported,
    ExplicitProgramConfig,
    ExplicitComputeKernelConfig,
    UnsupportedAttestation,
    AttestationQueryFailed,
    IncompleteRequest,
    CompatibilityUnavailable,
    CompatibilityMismatch,
    CircuitBroken,
    EmptyRegistry,
    ExactMiss,
    UnsupportedReplay,
    MaterializationRejected,
    CertifiedMatch,
    Count,
};

std::string_view resolution_reason_name(ResolutionReason reason) noexcept;

inline constexpr std::size_t kResolutionReasonCount = static_cast<std::size_t>(ResolutionReason::Count);

struct Eligibility {
    bool trace_capture_active = false;
    bool has_explicit_program_config = false;
    bool has_explicit_compute_kernel_config = false;
};

enum class AttestationStatus : std::uint8_t { Success, UnsupportedAttestation, QueryFailed };

struct AttestationResult {
    AttestationStatus status = AttestationStatus::UnsupportedAttestation;
    compact::DeviceDescriptor device{};
};

using AttestationProvider = AttestationResult (*)(const tt::tt_metal::distributed::MeshDevice&) noexcept;

// Multi-device promotion requires a reviewed canonical preimage containing
// ordered mesh coordinates, per-device capabilities, and fabric routing. The
// current public runtime API does not expose that complete contract, so the
// production provider explicitly fails closed instead of guessing fields.
AttestationResult production_attestation(const tt::tt_metal::distributed::MeshDevice&) noexcept;

struct RegistryRequest {
    compact::KeyDescriptor key{};

    bool operator==(const RegistryRequest&) const = default;
};

// Request construction stays separate from querying live TT objects. A future
// production provider must resolve every descriptor and digest before calling
// this pure, allocation-free seam; this function never invents a missing
// tensor, mesh, fabric, or runtime fact.
struct RegistryRequestFacts {
    Eligibility eligibility{};
    AttestationResult attestation{};
    compact::WorkloadDescriptor workload{};
    compact::OperationDescriptor operation{};
    compact::TensorDescriptor input{};
    compact::TensorDescriptor weight{};
    compact::OptionalTensorDescriptor bias{};
    compact::OptionalTensorDescriptor ternary_input_a{};
    compact::OptionalTensorDescriptor ternary_input_b{};
    compact::OptionalTensorDescriptor persistent_output{};
    compact::OptionalTensorDescriptor persistent_weight{};
};

enum class RequestBuildStatus : std::uint8_t {
    Success,
    TraceCaptureUnsupported,
    ExplicitProgramConfig,
    ExplicitComputeKernelConfig,
    UnsupportedAttestation,
    AttestationQueryFailed,
    IncompleteDescriptor,
    InconsistentDescriptor,
};

struct RequestBuildResult {
    RequestBuildStatus status = RequestBuildStatus::IncompleteDescriptor;
    std::optional<RegistryRequest> request = std::nullopt;
};

RequestBuildResult build_registry_request(const RegistryRequestFacts& facts) noexcept;

struct CompatibilityDigests {
    compact::Sha256 semantic_source_sha256{};
    compact::Sha256 build_identity_sha256{};
    compact::Sha256 runtime_capability_sha256{};
};

enum class CompatibilityStatus : std::uint8_t {
    Compatible,
    EmptyRegistry,
    Unavailable,
    SchemaMismatch,
    DigestMismatch,
};

CompatibilityStatus validate_compatibility(
    const compact::TableMetadata& metadata,
    std::size_t entry_count,
    const CompatibilityDigests& actual,
    const compact::DeviceDescriptor& device) noexcept;

struct Recipe {
    ttnn::experimental::prim::MinimalMatmulConfig config{};
    DeviceComputeKernelConfig compute_kernel_config{};
};

enum class MaterializationStatus : std::uint8_t {
    Success,
    UnsupportedSchema,
    InvalidProgramConfig,
    InvalidComputeKernelConfig,
};

struct MaterializationResult {
    MaterializationStatus status = MaterializationStatus::UnsupportedSchema;
    std::optional<Recipe> recipe = std::nullopt;
};

MaterializationResult materialize_recipe(const compact::EntryDescriptor& descriptor) noexcept;

struct Resolution {
    ResolutionReason reason = ResolutionReason::Disabled;
    const compact::EntryDescriptor* descriptor = nullptr;
};

enum class ExecutionAction : std::uint8_t { Fallback, ObserveOnly, ApplyRecipe };

struct DispatchResult {
    Resolution resolution{};
    ExecutionAction action = ExecutionAction::Fallback;
    std::optional<Recipe> recipe = std::nullopt;
};

ResolutionReason preflight(const Eligibility& eligibility) noexcept;
Resolution resolve(
    Mode mode,
    const std::optional<RegistryRequest>& request,
    AttestationStatus attestation_status,
    const Eligibility& eligibility) noexcept;

using ResolverFunction =
    Resolution (*)(Mode, const std::optional<RegistryRequest>&, AttestationStatus, const Eligibility&) noexcept;
using MaterializerFunction = MaterializationResult (*)(const compact::EntryDescriptor&);

// Off never calls the resolver. Shadow and On resolve at most once. Only On
// materializes, and any materialization failure circuit-breaks and falls back
// before the public device operation is attempted.
DispatchResult resolve_for_dispatch(
    Mode mode,
    const std::optional<RegistryRequest>& request,
    AttestationStatus attestation_status,
    const Eligibility& eligibility,
    ResolverFunction resolver = &resolve,
    MaterializerFunction materializer = &materialize_recipe) noexcept;

Resolution resolve_with_table_for_testing(
    Mode mode,
    const std::optional<RegistryRequest>& request,
    AttestationStatus attestation_status,
    const Eligibility& eligibility,
    const compact::TableMetadata& metadata,
    std::span<const compact::EntryDescriptor> entries,
    const CompatibilityDigests& actual) noexcept;

Mode current_mode() noexcept;
void reset_startup_mode_for_testing() noexcept;

bool circuit_break() noexcept;
bool is_circuit_broken() noexcept;
void reset_circuit_breaker_for_testing() noexcept;

struct StatsSnapshot {
    bool mode_is_frozen = false;
    Mode frozen_mode = Mode::Off;
    std::size_t entry_count = 0;
    std::uint64_t resolution_attempts = 0;
    std::uint64_t certified_hits = 0;
    std::uint64_t shadow_would_hits = 0;
    std::uint64_t selected_hits = 0;
    // The asynchronous public launch returned without throwing. This is not a
    // device synchronization or silicon-correctness signal.
    std::uint64_t launch_completed_hits = 0;
    std::uint64_t fallbacks = 0;
    std::uint64_t circuit_breaker_activations = 0;
    bool circuit_broken = false;
    std::array<std::uint64_t, kResolutionReasonCount> reasons{};
};

void record_resolution(Mode mode, const Resolution& resolution, ExecutionAction action) noexcept;
void record_launch_completed_hit() noexcept;
StatsSnapshot stats_snapshot() noexcept;
void reset_stats_for_testing() noexcept;

bool fail_closed_trace_capture_active(Mode mode, std::optional<bool> observed_active) noexcept;

class SelectedExecutionGuard {
public:
    explicit SelectedExecutionGuard(const bool* selected) noexcept;
    ~SelectedExecutionGuard() noexcept;

    SelectedExecutionGuard(const SelectedExecutionGuard&) = delete;
    SelectedExecutionGuard& operator=(const SelectedExecutionGuard&) = delete;

private:
    const bool* selected_;
    int uncaught_exceptions_;
};

// No fallback callable exists at this boundary. Once selected execution begins,
// an exception propagates and the guard circuit-breaks without a baseline retry.
template <typename Callable>
decltype(auto) execute_selected_call_once(const SelectedExecutionGuard&, Callable&& callable) {
    return std::forward<Callable>(callable)();
}

}  // namespace ttnn::experimental::all_gather_minimal_matmul_registry
