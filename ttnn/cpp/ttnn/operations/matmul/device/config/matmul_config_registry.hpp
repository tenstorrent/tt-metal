// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include <tt-metalium/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tile.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/matmul/device/config/registry/matmul_registry_attestation.hpp"
#include "ttnn/operations/matmul/device/config/registry/matmul_registry_descriptor.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/config.hpp"

namespace ttnn::operations::matmul::registry {

using Mode = ttnn::MatmulRegistryMode;

// The shared bound_matmul helper serves three public operations. The domain is
// explicit so a recipe certified for one operation can never match another.
enum class OperationDomain { DenseMatmul, Linear, Addmm, IneligibleSharedCaller };

struct CallSemantics {
    OperationDomain domain = OperationDomain::IneligibleSharedCaller;
    // Exact IEEE-754 binary32 spellings are part of addmm semantics. They stay
    // absent for every other domain and avoid lossy decimal normalization.
    std::optional<std::uint32_t> alpha_f32_bits = std::nullopt;
    std::optional<std::uint32_t> beta_f32_bits = std::nullopt;

    bool operator==(const CallSemantics&) const = default;
};

// Public wrappers use these constructors instead of spelling domains inline.
// Direct primitive, batched-weight, sparse, and CCL paths do not call them.
constexpr CallSemantics dense_matmul_call_semantics() noexcept {
    return CallSemantics{.domain = OperationDomain::DenseMatmul};
}
constexpr CallSemantics linear_call_semantics() noexcept { return CallSemantics{.domain = OperationDomain::Linear}; }
CallSemantics addmm_call_semantics(float alpha, float beta) noexcept;

enum class ResolutionReason {
    Disabled,
    IneligibleOperationDomain,
    MalformedOperationSemantics,
    InconsistentIoContract,
    TraceCaptureUnsupported,
    ExplicitOverride,
    UnsupportedSemantics,
    IncompleteRequest,
    InconsistentRequest,
    DeviceAttestationUnavailable,
    CompatibilityUninitialized,
    CompatibilitySchemaMismatch,
    SemanticSourceMismatch,
    BuildIdentityMismatch,
    RuntimeCapabilityMismatch,
    CircuitBroken,
    UnsupportedReplay,
    MaterializationRejected,
    EmptyRegistry,
    CertifiedMatch,
    Count,
};

std::string_view resolution_reason_name(ResolutionReason reason) noexcept;

// A failed inspector query is indistinguishable from active capture and must
// reject both Shadow and On. Off remains inert.
bool fail_closed_trace_capture_active(Mode mode, std::optional<bool> observed_active) noexcept;

inline constexpr std::size_t kOperationDomainCount = 4;
inline constexpr std::size_t kResolutionReasonCount = static_cast<std::size_t>(ResolutionReason::Count);

enum class IoContractStatus {
    Resolved,
    OptionalOutputMemoryMismatch,
    OptionalOutputDtypeMismatch,
    OutputTileConflict,
    InvalidTransposeTile,
};

struct OptionalOutputContract {
    tt::tt_metal::MemoryConfig memory_config;
    tt::tt_metal::DataType dtype;
    tt::tt_metal::Tile tile;
};

struct IoContractRequest {
    tt::tt_metal::DataType input_a_dtype;
    tt::tt_metal::Tile input_a_tile;
    tt::tt_metal::Tile input_b_tile;
    tt::tt_metal::MemoryConfig requested_output_memory_config;
    std::optional<tt::tt_metal::DataType> requested_output_dtype = std::nullopt;
    std::optional<tt::tt_metal::Tile> requested_output_tile = std::nullopt;
    std::optional<OptionalOutputContract> optional_output = std::nullopt;
    bool transpose_a = false;
    bool transpose_b = false;
};

struct ResolvedMatmulIoContract {
    IoContractStatus status;
    tt::tt_metal::MemoryConfig output_memory_config;
    tt::tt_metal::DataType output_dtype;
    tt::tt_metal::Tile output_tile;
    bool uses_optional_output = false;
};

// Resolve only caller-known I/O facts. Conflicts are returned as typed status so
// Shadow can observe without changing legacy validation order or exceptions.
ResolvedMatmulIoContract resolve_matmul_io_contract(const IoContractRequest& request);

struct Eligibility {
    CallSemantics call;
    IoContractStatus io_contract_status = IoContractStatus::Resolved;
    bool trace_capture_active = false;
    bool has_program_config = false;
    bool has_compute_kernel_config = false;
    bool has_user_core_grid = false;
    bool has_bias = false;
    bool has_activation = false;
    bool has_optional_output = false;
    bool has_output_tile = false;
    bool has_global_cb = false;
    bool has_sub_device = false;
    bool has_bcast_batch = false;
    bool untilize_out = false;
    bool input_a_sharded = false;
    bool input_b_sharded = false;
    bool output_sharded = false;
    bool input_b_batched = false;
    bool transpose_a = false;
    bool transpose_b = false;
};

// Shared caller-state projection used by both public wrapper dispatch and the
// read-only resolved-key inspector. Tests can compare wrapper defaults at this
// seam without querying a device or mutating registry state.
Eligibility v1_eligibility_from_call_state(
    CallSemantics call,
    IoContractStatus io_contract_status,
    bool trace_capture_active,
    bool has_bias,
    const ttnn::prim::MatmulParams& parameters,
    bool has_optional_output,
    bool input_a_sharded,
    bool input_b_sharded,
    bool output_sharded) noexcept;

// The generated table accepts this complete, exact request—not Eligibility.
// Keeping the lookup seam typed prevents a certified recipe from crossing an
// M/K/N, tensor contract, device, topology, or public-operation boundary.
struct TensorRequest {
    tt::tt_metal::DataType dtype;
    tt::tt_metal::Layout layout;
    tt::tt_metal::TensorMemoryLayout memory_layout;
    tt::tt_metal::BufferType buffer_type;
    std::uint32_t tile_height;
    std::uint32_t tile_width;

    bool operator==(const TensorRequest&) const = default;
};

struct WorkloadRequest {
    std::uint64_t logical_m;
    std::uint64_t logical_k;
    std::uint64_t logical_n;
    std::uint64_t padded_m;
    std::uint64_t padded_k;
    std::uint64_t padded_n;

    bool operator==(const WorkloadRequest&) const = default;
};

struct DeviceRequest {
    DeviceAttestationStatus attestation_status = DeviceAttestationStatus::QueryFailed;
    std::uint32_t architecture;
    std::uint32_t board_capability_class;
    std::uint32_t device_count;
    std::uint32_t mesh_rows;
    std::uint32_t mesh_cols;
    std::uint32_t compute_grid_x;
    std::uint32_t compute_grid_y;
    compact::Sha256 topology_sha256{};
    compact::Sha256 runtime_capability_sha256{};

    bool operator==(const DeviceRequest&) const = default;
};

struct MatmulRegistryRequest {
    static constexpr std::size_t kMaxActivationParameters = 8;
    std::uint32_t schema_version = 1;
    CallSemantics call;
    WorkloadRequest workload;
    TensorRequest input_a;
    TensorRequest input_b;
    TensorRequest output;
    DeviceRequest device;
    bool transpose_a;
    bool transpose_b;
    bool has_bias;
    bool has_activation;
    bool untilize_out;
    std::optional<bool> bcast_batch;
    bool run_batched;
    std::optional<std::uint32_t> activation_op;
    // Exact IEEE-754 binary32 spellings, in parameter order. Lookup requests
    // never own heap-backed containers.
    std::array<std::uint32_t, kMaxActivationParameters> activation_param_f32_bits{};
    std::uint8_t activation_param_count = 0;

    bool operator==(const MatmulRegistryRequest&) const = default;
};

// Lossless conversion into the immutable generated-table key. This is also
// the single source for the read-only measurement-worker key inspector; it
// performs no registry lookup, selection, initialization, or mutation.
std::optional<compact::KeyDescriptor> compact_registry_key(const MatmulRegistryRequest& request) noexcept;

struct RegistryRequestInspection {
    std::optional<MatmulRegistryRequest> request = std::nullopt;
    Eligibility eligibility{};
    DeviceAttestationResult device_attestation{};
};

// Build the exact request used by bound_matmul from live tensor and call-state
// facts. Tensor/device inspection may throw before legacy validation, so public
// dispatch keeps this call inside its observation-only catch boundary.
RegistryRequestInspection inspect_registry_request(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    bool has_bias,
    CallSemantics call_semantics,
    const ttnn::prim::MatmulParams& parameters,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    bool trace_capture_active,
    DeviceAttestationProvider provider = &production_device_attestation);

struct Recipe {
    MatmulProgramConfig program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    // Duplicated native state is selected explicitly rather than inferred from
    // one program-config alternative at the call site.
    bool untilize_out = false;
};

struct Resolution {
    ResolutionReason reason = ResolutionReason::Disabled;
    // Native pointer is retained only by the device-free synthetic seam.
    const Recipe* recipe = nullptr;
    // Generated static storage; never enters MatmulParams or its cache key.
    const compact::EntryDescriptor* descriptor = nullptr;
};

enum class MaterializationStatus {
    Success,
    UnsupportedSchema,
    UnsupportedReplay,
    InvalidProgramConfig,
    InvalidComputeKernelConfig,
    InvalidCallState,
};

struct MaterializationResult {
    MaterializationStatus status = MaterializationStatus::UnsupportedReplay;
    std::optional<Recipe> recipe = std::nullopt;
};

// Converts compact immutable data into one complete native recipe. This may
// allocate in future replay families and therefore intentionally is not noexcept.
MaterializationResult materialize_matmul_registry_recipe(const compact::EntryDescriptor& descriptor);

struct CompatibilityDigests {
    compact::Sha256 semantic_source_sha256{};
    compact::Sha256 build_identity_sha256{};
    compact::Sha256 runtime_capability_sha256{};
};

inline constexpr std::string_view kCompatibilityAttestationArtifactKind = "ttnn_matmul_registry_runtime_attestation";
inline constexpr std::uint16_t kCompatibilityAttestationSchemaVersion = 1;

// Stable read-only attestation surface for measurement workers. A worker must
// reject the report unless device_attestation_status is Success; digest fields
// that depend on the device are zero on failure.
struct RegistryCompatibilityAttestation {
    std::uint16_t schema_version = kCompatibilityAttestationSchemaVersion;
    DeviceAttestationStatus device_attestation_status = DeviceAttestationStatus::QueryFailed;
    std::uint16_t codegen_recipe_abi = compact::kCodegenRecipeAbi;
    std::uint32_t board_capability_class = 0;
    compact::Sha256 actual_semantic_source_sha256{};
    compact::Sha256 actual_build_identity_sha256{};
    compact::Sha256 actual_topology_sha256{};
    compact::Sha256 actual_runtime_capability_sha256{};
};

enum class CompatibilityStatus {
    Uninitialized,
    EmptyRegistry,
    Compatible,
    SchemaMismatch,
    SemanticSourceMismatch,
    BuildIdentityMismatch,
    RuntimeCapabilityMismatch,
};

CompatibilityStatus validate_registry_compatibility(
    const compact::TableMetadata& expected, std::size_t entry_count, const CompatibilityDigests& actual) noexcept;

// Explicit process-start initialization. The first caller freezes the result;
// later calls observe it and cannot replace the attestation.
CompatibilityStatus initialize_registry_compatibility(const CompatibilityDigests& actual) noexcept;
CompatibilityStatus startup_compatibility_status() noexcept;
CompatibilityDigests compiled_registry_compatibility_digests(const compact::Sha256& runtime_capability_sha256) noexcept;
CompatibilityStatus initialize_registry_compatibility_from_attestation(
    const DeviceAttestationResult& attestation) noexcept;
RegistryCompatibilityAttestation registry_compatibility_attestation(
    const DeviceAttestationResult& attestation) noexcept;
RegistryCompatibilityAttestation query_registry_compatibility_attestation(
    const tt::tt_metal::distributed::MeshDevice& device,
    DeviceAttestationProvider provider = &production_device_attestation) noexcept;

// Test only: no concurrent registry dispatch may be active.
CompatibilityStatus initialize_registry_compatibility_for_testing(
    const compact::TableMetadata& expected, std::size_t entry_count, const CompatibilityDigests& actual) noexcept;
void reset_startup_compatibility_for_testing() noexcept;

enum class ExecutionAction { Fallback, ObserveOnly, ApplyRecipe };

// A certified Shadow hit is observable but never mutates execution parameters.
ExecutionAction execution_action(Mode mode, const Resolution& resolution) noexcept;

// Construct one complete temporary parameter object before dispatch. Shadow
// and fallback return nullopt without copying or mutating the legacy object.
// Native recipe copies can allocate. The dispatch gate catches construction
// failures, circuit-breaks the affected domain, and preserves the legacy path.
std::optional<ttnn::prim::MatmulParams> materialize_parameters_for_execution(
    Mode mode, const Resolution& resolution, const ttnn::prim::MatmulParams& legacy_parameters);

// Every recipe carries one effective untilize_out value. It must agree with the
// duplicated field in the 1D config, and must be false for all other families.
bool has_consistent_untilize_out(const Recipe& recipe) noexcept;

// Snapshots CONFIG on first use. Subsequent configuration mutation cannot alter
// behavior during the process lifetime.
Mode current_mode() noexcept;

// Test only: callers must ensure no concurrent matmul dispatch is in flight.
void reset_startup_mode_for_testing() noexcept;

// Device-free admission plus exact lookup in the generated immutable table.
// CertifiedMatch means only that the request is inside the exact v1 envelope;
// it performs no lookup, compatibility initialization, or telemetry mutation.
ResolutionReason preflight_v1_eligibility(const Eligibility& eligibility) noexcept;
ResolutionReason validate_v1_request_envelope(
    const MatmulRegistryRequest& request, const Eligibility& eligibility) noexcept;
Resolution resolve(Mode mode, const MatmulRegistryRequest& request, const Eligibility& eligibility) noexcept;

using ResolverFunction = Resolution (*)(Mode, const MatmulRegistryRequest&, const Eligibility&) noexcept;
using MaterializerFunction = MaterializationResult (*)(const compact::EntryDescriptor&);

struct DispatchResult {
    Resolution resolution;
    ExecutionAction action = ExecutionAction::Fallback;
    std::optional<ttnn::prim::MatmulParams> materialized_parameters = std::nullopt;
};

// The single dispatch gate. Off never calls the resolver. Shadow and On call it
// at most once, and only On may produce a separate materialized parameter set.
DispatchResult resolve_for_dispatch(
    Mode mode,
    const std::optional<MatmulRegistryRequest>& request,
    const Eligibility& eligibility,
    const ttnn::prim::MatmulParams& legacy_parameters,
    ResolverFunction resolver = &resolve,
    MaterializerFunction materializer = &materialize_matmul_registry_recipe);

// Device-free synthetic-hit seam used to prove exact-key, Shadow, and On
// behavior while the generated production table remains empty.
Resolution resolve_with_synthetic_candidate_for_testing(
    Mode mode,
    const MatmulRegistryRequest& request,
    const Eligibility& eligibility,
    const MatmulRegistryRequest& candidate_request,
    const Recipe& candidate_recipe) noexcept;

Resolution resolve_with_compact_table_for_testing(
    Mode mode,
    const MatmulRegistryRequest& request,
    const Eligibility& eligibility,
    const compact::TableMetadata& metadata,
    std::span<const compact::EntryDescriptor> entries,
    const CompatibilityDigests& actual) noexcept;

bool circuit_break_domain(OperationDomain domain) noexcept;
bool is_domain_circuit_broken(OperationDomain domain) noexcept;

// Test only: no concurrent registry dispatch may be active.
void reset_circuit_breakers_for_testing() noexcept;

struct DomainStatsSnapshot {
    std::uint64_t resolution_attempts = 0;
    std::uint64_t certified_hits = 0;
    std::uint64_t shadow_would_hits = 0;
    std::uint64_t selected_hits = 0;
    std::uint64_t completed_hits = 0;
    std::uint64_t fallbacks = 0;
    std::uint64_t circuit_breaker_activations = 0;
    bool circuit_broken = false;
    std::array<std::uint64_t, kResolutionReasonCount> reasons{};
};

struct StatsSnapshot {
    bool mode_is_frozen = false;
    Mode frozen_mode = Mode::Off;
    CompatibilityStatus compatibility_status = CompatibilityStatus::Uninitialized;
    compact::TableMetadata table_metadata{};
    std::size_t entry_count = 0;
    std::array<DomainStatsSnapshot, kOperationDomainCount> domains{};
};

// Telemetry cardinality is bounded by fixed domain and reason enums. Request
// dimensions and identifiers are deliberately never retained.
void record_resolution(
    Mode mode, OperationDomain domain, const Resolution& resolution, ExecutionAction action) noexcept;
void record_completed_hit(OperationDomain domain) noexcept;
StatsSnapshot stats_snapshot() noexcept;

// Public wrapper scope guard. A selected operation records completion only
// after its full public post-processing path returns. An exception propagates
// unchanged, circuit-breaks the domain, and is never retried with a baseline
// configuration inside the same public call.
class SelectedExecutionGuard {
public:
    SelectedExecutionGuard(OperationDomain domain, const bool* selected) noexcept;
    ~SelectedExecutionGuard() noexcept;

    SelectedExecutionGuard(const SelectedExecutionGuard&) = delete;
    SelectedExecutionGuard& operator=(const SelectedExecutionGuard&) = delete;
    SelectedExecutionGuard(SelectedExecutionGuard&&) = delete;
    SelectedExecutionGuard& operator=(SelectedExecutionGuard&&) = delete;

private:
    OperationDomain domain_;
    const bool* selected_;
    int uncaught_exceptions_;
};

// The public wrappers hand their selected execution to this one-shot boundary.
// There is intentionally no fallback callable: once a certified recipe is
// selected, an execution error propagates and the surrounding guard
// circuit-breaks the domain instead of retrying the public operation.
template <typename Callable>
decltype(auto) execute_selected_call_once(const SelectedExecutionGuard&, Callable&& callable) {
    return std::forward<Callable>(callable)();
}

// Test only: callers must ensure no concurrent recording is in flight.
void reset_stats_for_testing() noexcept;

}  // namespace ttnn::operations::matmul::registry
