// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/matmul_config_registry.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <exception>
#include <limits>
#include <utility>

#include "matmul_registry_build_attestation.hpp"
#include "matmul_registry_data.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::matmul::registry {
namespace {

constexpr std::uint8_t kModeUninitialized = 0xff;
constexpr std::uint8_t kCompatibilityUninitialized = 0xff;
constexpr std::uint8_t kCompatibilityInitializing = 0xfe;
std::atomic<std::uint8_t> frozen_mode{kModeUninitialized};
std::atomic<std::uint8_t> frozen_compatibility{kCompatibilityUninitialized};

struct AtomicDomainStats {
    std::atomic<std::uint64_t> resolution_attempts{0};
    std::atomic<std::uint64_t> certified_hits{0};
    std::atomic<std::uint64_t> shadow_would_hits{0};
    std::atomic<std::uint64_t> selected_hits{0};
    std::atomic<std::uint64_t> completed_hits{0};
    std::atomic<std::uint64_t> fallbacks{0};
    std::atomic<std::uint64_t> circuit_breaker_activations{0};
    std::array<std::atomic<std::uint64_t>, kResolutionReasonCount> reasons{};
};

std::array<AtomicDomainStats, kOperationDomainCount> stats;
std::array<std::atomic<bool>, kOperationDomainCount> circuit_breakers{};
std::array<std::atomic<std::uint64_t>, kDistributedMatmulClassCount> distributed_observations{};

constexpr std::size_t index(const OperationDomain domain) { return static_cast<std::size_t>(domain); }
constexpr std::size_t index(const ResolutionReason reason) { return static_cast<std::size_t>(reason); }
constexpr std::size_t index(const DistributedMatmulClass classification) {
    return static_cast<std::size_t>(classification);
}

bool spans_equal(const std::span<const std::uint32_t> lhs, const std::span<const std::uint32_t> rhs) noexcept {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool coordinate_equal(
    const tt::tt_metal::distributed::MeshCoordinate& lhs,
    const tt::tt_metal::distributed::MeshCoordinate& rhs) noexcept {
    const auto lhs_coords = lhs.coords();
    const auto rhs_coords = rhs.coords();
    return lhs_coords.size() == rhs_coords.size() &&
           std::equal(lhs_coords.begin(), lhs_coords.end(), rhs_coords.begin());
}

bool coordinate_spans_equal(
    const std::span<const tt::tt_metal::distributed::MeshCoordinate> lhs,
    const std::span<const tt::tt_metal::distributed::MeshCoordinate> rhs) noexcept {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (!coordinate_equal(lhs[i], rhs[i])) {
            return false;
        }
    }
    return true;
}

bool is_exact_bh32_coordinate_order(
    const std::span<const tt::tt_metal::distributed::MeshCoordinate> coordinates) noexcept {
    constexpr std::size_t rows = 8;
    constexpr std::size_t cols = 4;
    if (coordinates.size() != rows * cols) {
        return false;
    }
    for (std::size_t i = 0; i < coordinates.size(); ++i) {
        const auto values = coordinates[i].coords();
        if (values.size() != 2 || values[0] != i / cols || values[1] != i % cols) {
            return false;
        }
    }
    return true;
}

bool is_exact_bh32_tensor_coordinates(const DistributedTensorView& tensor) noexcept {
    return is_exact_bh32_coordinate_order(tensor.mesh_coordinates) &&
           coordinate_spans_equal(tensor.mesh_coordinates, tensor.storage_coordinates);
}

bool is_replicate(const tt::tt_metal::distributed::MeshMapperConfig::Placement& placement) noexcept {
    return std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Replicate>(placement);
}

bool is_shard(const tt::tt_metal::distributed::MeshMapperConfig::Placement& placement, const int tensor_dim) noexcept {
    const auto* shard = std::get_if<tt::tt_metal::distributed::MeshMapperConfig::Shard>(&placement);
    return shard != nullptr && shard->dim == tensor_dim;
}

bool is_exact_8x4_distribution(const DistributedTensorView& tensor) noexcept {
    constexpr std::array<std::uint32_t, 2> expected{8, 4};
    return spans_equal(tensor.distribution_shape, expected);
}

bool is_exact_replicated_32_distribution(const DistributedTensorView& tensor) noexcept {
    constexpr std::array<std::uint32_t, 1> expected{32};
    return spans_equal(tensor.distribution_shape, expected) && tensor.placements.size() == 1 &&
           is_replicate(tensor.placements[0]);
}

bool has_exact_local_tensor_contract(const DistributedTensorView& tensor, const bool is_input_a) noexcept {
    const bool supported_dtype = is_input_a ? tensor.dtype == tt::tt_metal::DataType::BFLOAT16
                                            : tensor.dtype == tt::tt_metal::DataType::BFLOAT16 ||
                                                  tensor.dtype == tt::tt_metal::DataType::BFLOAT8_B ||
                                                  tensor.dtype == tt::tt_metal::DataType::BFLOAT4_B;
    return supported_dtype && tensor.layout == tt::tt_metal::Layout::TILE &&
           tensor.memory_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
           tensor.buffer_type == tt::tt_metal::BufferType::DRAM;
}

bool has_exact_bare_matmul_call_contract(const DistributedMatmulObservation& observation) noexcept {
    return observation.domain == OperationDomain::DenseMatmul && observation.tensors_share_device &&
           observation.device_count == 32 && observation.device_mesh_shape.size() == 2 &&
           observation.device_mesh_shape[0] == 8 && observation.device_mesh_shape[1] == 4 && !observation.transpose_a &&
           !observation.transpose_b && !observation.has_bias && !observation.has_activation &&
           !observation.has_program_config && !observation.has_compute_kernel_config &&
           !observation.has_user_core_grid && !observation.has_output_dtype && !observation.has_optional_output &&
           !observation.has_output_tile && !observation.has_global_cb && !observation.has_sub_device &&
           !observation.has_bcast_batch && !observation.untilize_out && !observation.run_batched &&
           observation.output_is_dram_interleaved && has_exact_local_tensor_contract(observation.input_a, true) &&
           has_exact_local_tensor_contract(observation.input_b, false) &&
           is_exact_bh32_tensor_coordinates(observation.input_a) &&
           is_exact_bh32_tensor_coordinates(observation.input_b) &&
           coordinate_spans_equal(observation.input_a.mesh_coordinates, observation.input_b.mesh_coordinates);
}

bool is_exact_dp_rank_distinct_v1(const DistributedMatmulObservation& observation) noexcept {
    const auto& a = observation.input_a;
    const auto& b = observation.input_b;
    return a.logical_shape.size() == 4 && a.logical_shape[0] == 1 && a.logical_shape[1] == 1 &&
           a.logical_shape[2] > 0 && a.logical_shape[3] > 0 && b.logical_shape.size() == 2 &&
           b.logical_shape[0] == a.logical_shape[3] && b.logical_shape[1] > 0 && is_exact_8x4_distribution(a) &&
           a.placements.size() == 2 && is_shard(a.placements[0], 0) && is_shard(a.placements[1], 1) &&
           is_exact_replicated_32_distribution(b);
}

bool is_exact_tpn_sp_m_tp_n_v1(const DistributedMatmulObservation& observation) noexcept {
    const auto& a = observation.input_a;
    const auto& b = observation.input_b;
    return a.logical_shape.size() == 2 && a.logical_shape[0] > 0 && a.logical_shape[1] > 0 &&
           b.logical_shape.size() == 2 && b.logical_shape[0] == a.logical_shape[1] && b.logical_shape[1] > 0 &&
           is_exact_8x4_distribution(a) && is_exact_8x4_distribution(b) && a.placements.size() == 2 &&
           b.placements.size() == 2 && is_shard(a.placements[0], 0) && is_replicate(a.placements[1]) &&
           is_replicate(b.placements[0]) && is_shard(b.placements[1], 1);
}

ResolutionReason compatibility_reason(const CompatibilityStatus status) noexcept {
    switch (status) {
        case CompatibilityStatus::Uninitialized: return ResolutionReason::CompatibilityUninitialized;
        case CompatibilityStatus::SchemaMismatch: return ResolutionReason::CompatibilitySchemaMismatch;
        case CompatibilityStatus::SemanticSourceMismatch: return ResolutionReason::SemanticSourceMismatch;
        case CompatibilityStatus::BuildIdentityMismatch: return ResolutionReason::BuildIdentityMismatch;
        case CompatibilityStatus::RuntimeCapabilityMismatch: return ResolutionReason::RuntimeCapabilityMismatch;
        case CompatibilityStatus::EmptyRegistry: return ResolutionReason::EmptyRegistry;
        case CompatibilityStatus::Compatible: return ResolutionReason::CertifiedMatch;
    }
    return ResolutionReason::CompatibilitySchemaMismatch;
}

std::optional<compact::DataType> compact_dtype(const tt::tt_metal::DataType dtype) noexcept {
    switch (dtype) {
        case tt::tt_metal::DataType::BFLOAT16: return compact::DataType::BFloat16;
        case tt::tt_metal::DataType::FLOAT32: return compact::DataType::Float32;
        case tt::tt_metal::DataType::BFLOAT8_B: return compact::DataType::BFloat8B;
        default: return std::nullopt;
    }
}

std::optional<compact::Layout> compact_layout(const tt::tt_metal::Layout layout) noexcept {
    switch (layout) {
        case tt::tt_metal::Layout::TILE: return compact::Layout::Tile;
        case tt::tt_metal::Layout::ROW_MAJOR: return compact::Layout::RowMajor;
        default: return std::nullopt;
    }
}

std::optional<compact::BufferType> compact_buffer_type(const tt::tt_metal::BufferType buffer_type) noexcept {
    switch (buffer_type) {
        case tt::tt_metal::BufferType::DRAM: return compact::BufferType::Dram;
        case tt::tt_metal::BufferType::L1: return compact::BufferType::L1;
        default: return std::nullopt;
    }
}

std::optional<compact::TensorDescriptor> compact_tensor(const TensorRequest& tensor) noexcept {
    const auto dtype = compact_dtype(tensor.dtype);
    const auto layout = compact_layout(tensor.layout);
    const auto buffer_type = compact_buffer_type(tensor.buffer_type);
    if (!dtype.has_value() || !layout.has_value() || !buffer_type.has_value() ||
        tensor.memory_layout != tt::tt_metal::TensorMemoryLayout::INTERLEAVED ||
        tensor.tile_height > std::numeric_limits<std::uint16_t>::max() ||
        tensor.tile_width > std::numeric_limits<std::uint16_t>::max()) {
        return std::nullopt;
    }
    return compact::TensorDescriptor{
        .buffer_type = *buffer_type,
        .dtype = *dtype,
        .layout = *layout,
        .memory_layout = compact::MemoryLayout::Interleaved,
        .tile_height = static_cast<std::uint16_t>(tensor.tile_height),
        .tile_width = static_cast<std::uint16_t>(tensor.tile_width)};
}

std::optional<compact::Domain> compact_domain(const OperationDomain domain) noexcept {
    switch (domain) {
        case OperationDomain::DenseMatmul: return compact::Domain::DenseMatmul;
        case OperationDomain::Linear: return compact::Domain::DenseLinear;
        case OperationDomain::Addmm: return compact::Domain::DenseAddmm;
        case OperationDomain::IneligibleSharedCaller: return std::nullopt;
    }
    return std::nullopt;
}

std::optional<compact::KeyDescriptor> compact_registry_key_impl(const MatmulRegistryRequest& request) noexcept {
    const auto input_a = compact_tensor(request.input_a);
    const auto input_b = compact_tensor(request.input_b);
    const auto output = compact_tensor(request.output);
    const auto domain = compact_domain(request.call.domain);
    const auto& device = request.device;
    if (!input_a.has_value() || !input_b.has_value() || !output.has_value() || !domain.has_value() ||
        device.device_count > std::numeric_limits<std::uint16_t>::max() ||
        device.mesh_rows > std::numeric_limits<std::uint16_t>::max() ||
        device.mesh_cols > std::numeric_limits<std::uint16_t>::max() ||
        device.compute_grid_x > std::numeric_limits<std::uint16_t>::max() ||
        device.compute_grid_y > std::numeric_limits<std::uint16_t>::max()) {
        return std::nullopt;
    }

    return compact::KeyDescriptor{
        .architecture = device.architecture,
        .bcast_batch_present = request.bcast_batch.has_value(),
        .bcast_batch = request.bcast_batch.value_or(false),
        .board_capability_class = device.board_capability_class,
        .codegen_recipe_abi = compact::kCodegenRecipeAbi,
        .compute_grid_x = static_cast<std::uint16_t>(device.compute_grid_x),
        .compute_grid_y = static_cast<std::uint16_t>(device.compute_grid_y),
        .device_count = static_cast<std::uint16_t>(device.device_count),
        .has_activation = request.has_activation,
        .has_bias = request.has_bias,
        .input_a = *input_a,
        .input_b = *input_b,
        .logical_k = request.workload.logical_k,
        .logical_m = request.workload.logical_m,
        .logical_n = request.workload.logical_n,
        .mesh_cols = static_cast<std::uint16_t>(device.mesh_cols),
        .mesh_rows = static_cast<std::uint16_t>(device.mesh_rows),
        .output = *output,
        .padded_k = request.workload.padded_k,
        .padded_m = request.workload.padded_m,
        .padded_n = request.workload.padded_n,
        .run_batched = request.run_batched,
        .schema_version = static_cast<std::uint16_t>(request.schema_version),
        .topology_sha256 = device.topology_sha256,
        .transpose_a = request.transpose_a,
        .transpose_b = request.transpose_b,
        .untilize_out = request.untilize_out,
        .domain = *domain,
        .alpha_f32_bits = request.call.alpha_f32_bits.value_or(0),
        .beta_f32_bits = request.call.beta_f32_bits.value_or(0)};
}

bool has_valid_multi_core_reuse_work_split(
    const compact::KeyDescriptor& key, const compact::MultiCoreReuseDescriptor& program) noexcept {
    const auto& input_a = key.input_a;
    const auto& input_b = key.input_b;
    const auto& output = key.output;
    if (key.logical_m == 0 || key.logical_k == 0 || key.logical_n == 0 || key.padded_m < key.logical_m ||
        key.padded_k < key.logical_k || key.padded_n < key.logical_n || input_a.tile_height == 0 ||
        input_a.tile_width == 0 || input_b.tile_height == 0 || input_b.tile_width == 0 || output.tile_height == 0 ||
        output.tile_width == 0 || input_a.tile_height != output.tile_height ||
        input_b.tile_width != output.tile_width || key.padded_m % input_a.tile_height != 0 ||
        key.padded_k % input_a.tile_width != 0 || key.padded_k % input_b.tile_height != 0 ||
        key.padded_n % input_b.tile_width != 0) {
        return false;
    }

    const auto m_tiles = key.padded_m / input_a.tile_height;
    const auto input_a_k_tiles = key.padded_k / input_a.tile_width;
    const auto input_b_k_tiles = key.padded_k / input_b.tile_height;
    const auto n_tiles = key.padded_n / input_b.tile_width;
    return input_a_k_tiles == input_b_k_tiles && input_a_k_tiles % program.in0_block_w == 0 &&
           m_tiles % program.per_core_m == 0 && n_tiles == program.per_core_n &&
           program.per_core_m % program.out_subblock_h == 0 && program.per_core_n % program.out_subblock_w == 0;
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

std::string_view distributed_matmul_class_name(const DistributedMatmulClass classification) noexcept {
    switch (classification) {
        case DistributedMatmulClass::NotDistributed: return "not_distributed";
        case DistributedMatmulClass::DpRankDistinctV1: return "distributed.dp.rank_distinct_v1";
        case DistributedMatmulClass::TpnSpMTpNV1: return "distributed.tpn.sp_m_tp_n_v1";
        case DistributedMatmulClass::Unknown: return "distributed.unknown";
        case DistributedMatmulClass::Count: return "count";
    }
    return "distributed.unknown";
}

DistributedMatmulClass classify_distributed_matmul(const DistributedMatmulObservation& observation) noexcept {
    if (observation.device_count <= 1) {
        return DistributedMatmulClass::NotDistributed;
    }
    if (!has_exact_bare_matmul_call_contract(observation)) {
        return DistributedMatmulClass::Unknown;
    }
    if (is_exact_dp_rank_distinct_v1(observation)) {
        return DistributedMatmulClass::DpRankDistinctV1;
    }
    if (is_exact_tpn_sp_m_tp_n_v1(observation)) {
        return DistributedMatmulClass::TpnSpMTpNV1;
    }
    return DistributedMatmulClass::Unknown;
}

std::string_view resolution_reason_name(const ResolutionReason reason) noexcept {
    switch (reason) {
        case ResolutionReason::Disabled: return "disabled";
        case ResolutionReason::IneligibleOperationDomain: return "ineligible_operation_domain";
        case ResolutionReason::MalformedOperationSemantics: return "malformed_operation_semantics";
        case ResolutionReason::InconsistentIoContract: return "inconsistent_io_contract";
        case ResolutionReason::TraceCaptureUnsupported: return "trace_capture_unsupported";
        case ResolutionReason::ExplicitOverride: return "explicit_override";
        case ResolutionReason::UnsupportedSemantics: return "unsupported_semantics";
        case ResolutionReason::IncompleteRequest: return "incomplete_request";
        case ResolutionReason::InconsistentRequest: return "inconsistent_request";
        case ResolutionReason::DeviceAttestationUnavailable: return "device_attestation_unavailable";
        case ResolutionReason::CompatibilityUninitialized: return "compatibility_uninitialized";
        case ResolutionReason::CompatibilitySchemaMismatch: return "compatibility_schema_mismatch";
        case ResolutionReason::SemanticSourceMismatch: return "semantic_source_mismatch";
        case ResolutionReason::BuildIdentityMismatch: return "build_identity_mismatch";
        case ResolutionReason::RuntimeCapabilityMismatch: return "runtime_capability_mismatch";
        case ResolutionReason::CircuitBroken: return "circuit_broken";
        case ResolutionReason::UnsupportedReplay: return "unsupported_replay";
        case ResolutionReason::MaterializationRejected: return "materialization_rejected";
        case ResolutionReason::EmptyRegistry: return "empty_registry";
        case ResolutionReason::CertifiedMatch: return "certified_match";
        case ResolutionReason::PredictedMatch: return "predicted_match";
        case ResolutionReason::Count: return "count";
    }
    return "unknown";
}

bool fail_closed_trace_capture_active(const Mode mode, const std::optional<bool> observed_active) noexcept {
    return mode != Mode::Off && observed_active.value_or(true);
}

std::optional<compact::KeyDescriptor> compact_registry_key(const MatmulRegistryRequest& request) noexcept {
    return compact_registry_key_impl(request);
}

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

CompatibilityStatus validate_registry_compatibility(
    const compact::TableMetadata& expected,
    const std::size_t entry_count,
    const CompatibilityDigests& actual) noexcept {
    if (expected.lock_schema_version != 1 || expected.key_schema_version != 1 || expected.replay_schema_version != 2) {
        return CompatibilityStatus::SchemaMismatch;
    }
    if (entry_count == 0) {
        return CompatibilityStatus::EmptyRegistry;
    }
    if (expected.semantic_source_sha256 != actual.semantic_source_sha256) {
        return CompatibilityStatus::SemanticSourceMismatch;
    }
    const bool direct_bank_scope = expected.program_config_only_evidence_schema_version == 2 ||
                                   expected.online_program_config_model_evidence_schema_version == 2;
    if (!direct_bank_scope && expected.build_identity_sha256 != actual.build_identity_sha256) {
        return CompatibilityStatus::BuildIdentityMismatch;
    }
    // Direct-bank evidence (schema 2) binds source semantics. A checked lock
    // has no truthful build-local or per-session capability digest to bind.
    if (!direct_bank_scope && expected.runtime_capability_sha256 != actual.runtime_capability_sha256) {
        return CompatibilityStatus::RuntimeCapabilityMismatch;
    }
    return CompatibilityStatus::Compatible;
}

CompatibilityStatus initialize_registry_compatibility_for_testing(
    const compact::TableMetadata& expected,
    const std::size_t entry_count,
    const CompatibilityDigests& actual) noexcept {
    auto state = frozen_compatibility.load(std::memory_order_acquire);
    if (state != kCompatibilityUninitialized && state != kCompatibilityInitializing) {
        return static_cast<CompatibilityStatus>(state);
    }

    if (state == kCompatibilityUninitialized &&
        frozen_compatibility.compare_exchange_strong(
            state, kCompatibilityInitializing, std::memory_order_acq_rel, std::memory_order_acquire)) {
        const auto status = validate_registry_compatibility(expected, entry_count, actual);
        frozen_compatibility.store(static_cast<std::uint8_t>(status), std::memory_order_release);
        return status;
    }

    do {
        state = frozen_compatibility.load(std::memory_order_acquire);
    } while (state == kCompatibilityInitializing);
    return static_cast<CompatibilityStatus>(state);
}

CompatibilityStatus initialize_registry_compatibility(const CompatibilityDigests& actual) noexcept {
    std::size_t model_candidate_count = generated::program_config_exact_entries().size();
    for (const auto& model : generated::online_models()) {
        model_candidate_count += model.candidates.size();
    }
    return initialize_registry_compatibility_for_testing(
        generated::metadata(), generated::entries().size() + model_candidate_count, actual);
}

CompatibilityStatus startup_compatibility_status() noexcept {
    const auto state = frozen_compatibility.load(std::memory_order_acquire);
    return state == kCompatibilityUninitialized || state == kCompatibilityInitializing
               ? CompatibilityStatus::Uninitialized
               : static_cast<CompatibilityStatus>(state);
}

void reset_startup_compatibility_for_testing() noexcept {
    frozen_compatibility.store(kCompatibilityUninitialized, std::memory_order_release);
}

CompatibilityDigests compiled_registry_compatibility_digests(
    const compact::Sha256& runtime_capability_sha256) noexcept {
    static_assert(generated_build::kAttestationSchemaVersion == 1);
    return CompatibilityDigests{
        .semantic_source_sha256 = generated_build::kActualSemanticSourceSha256,
        .build_identity_sha256 = generated_build::kActualBuildIdentitySha256,
        .runtime_capability_sha256 = runtime_capability_sha256};
}

CompatibilityStatus initialize_registry_compatibility_from_attestation(
    const DeviceAttestationResult& attestation) noexcept {
    if (attestation.status != DeviceAttestationStatus::Success) {
        return startup_compatibility_status();
    }
    return initialize_registry_compatibility(
        compiled_registry_compatibility_digests(attestation.attestation.runtime_capability_sha256));
}

RegistryCompatibilityAttestation registry_compatibility_attestation(
    const DeviceAttestationResult& attestation) noexcept {
    const auto actual = compiled_registry_compatibility_digests(attestation.attestation.runtime_capability_sha256);
    return RegistryCompatibilityAttestation{
        .device_attestation_status = attestation.status,
        .board_capability_class = attestation.attestation.board_capability_class,
        .actual_semantic_source_sha256 = actual.semantic_source_sha256,
        .actual_build_identity_sha256 = actual.build_identity_sha256,
        .actual_topology_sha256 = attestation.attestation.topology_sha256,
        .actual_runtime_capability_sha256 = actual.runtime_capability_sha256};
}

RegistryCompatibilityAttestation query_registry_compatibility_attestation(
    const tt::tt_metal::distributed::MeshDevice& device, const DeviceAttestationProvider provider) noexcept {
    return registry_compatibility_attestation(query_device_attestation(device, provider));
}

CallSemantics addmm_call_semantics(const float alpha, const float beta) noexcept {
    return CallSemantics{
        .domain = OperationDomain::Addmm,
        .alpha_f32_bits = std::bit_cast<std::uint32_t>(alpha),
        .beta_f32_bits = std::bit_cast<std::uint32_t>(beta)};
}

bool has_nondefault_v1_tile_transpose(const tt::tt_metal::Tile& tile) noexcept {
    return tile.get_transpose_of_faces() || tile.get_transpose_within_face();
}

Eligibility v1_eligibility_from_call_state(
    const CallSemantics call,
    const IoContractStatus io_contract_status,
    const bool trace_capture_active,
    const bool has_bias,
    const ttnn::prim::MatmulParams& parameters,
    const bool has_optional_output,
    const bool input_a_sharded,
    const bool input_b_sharded,
    const bool output_sharded,
    const bool has_unsupported_tile_metadata) noexcept {
    return Eligibility{
        .call = call,
        .io_contract_status = io_contract_status,
        .trace_capture_active = trace_capture_active,
        .has_program_config = parameters.program_config.has_value(),
        .has_compute_kernel_config = parameters.compute_kernel_config.has_value(),
        .has_user_core_grid = parameters.user_core_coord.has_value(),
        .has_bias = has_bias,
        .has_activation = parameters.user_fused_activation.has_value(),
        .has_optional_output = has_optional_output,
        .has_output_tile = parameters.output_tile.has_value(),
        .has_global_cb = parameters.global_cb.has_value(),
        .has_sub_device = parameters.sub_device_id.has_value(),
        .has_bcast_batch = parameters.bcast_batch.has_value(),
        .untilize_out = parameters.untilize_out,
        .input_a_sharded = input_a_sharded,
        .input_b_sharded = input_b_sharded,
        .output_sharded = output_sharded,
        .input_b_batched = parameters.user_run_batched,
        .transpose_a = parameters.transpose_a,
        .transpose_b = parameters.transpose_b,
        .has_unsupported_tile_metadata = has_unsupported_tile_metadata};
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
    const bool has_native_recipe = resolution.recipe != nullptr && has_consistent_untilize_out(*resolution.recipe);
    const bool has_compact_recipe = resolution.descriptor != nullptr;
    const bool has_predicted_config =
        resolution.predicted_program_config.has_value() && resolution.predicted_key.has_value();
    const bool selected =
        resolution.reason == ResolutionReason::CertifiedMatch || resolution.reason == ResolutionReason::PredictedMatch;
    if (!selected || (!has_native_recipe && !has_compact_recipe && !has_predicted_config)) {
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

    if (resolution.predicted_program_config.has_value() && resolution.predicted_key.has_value()) {
        auto program_config =
            materialize_registry_program_config(*resolution.predicted_key, *resolution.predicted_program_config);
        if (!program_config.has_value()) {
            return std::nullopt;
        }
        auto materialized = legacy_parameters;
        materialized.program_config = std::move(program_config);
        return materialized;
    }

    std::optional<Recipe> compact_recipe;
    const Recipe* selected_recipe = resolution.recipe;
    if (resolution.descriptor != nullptr) {
        auto result = materialize_matmul_registry_recipe(*resolution.descriptor);
        if (result.status != MaterializationStatus::Success || !result.recipe.has_value()) {
            return std::nullopt;
        }
        compact_recipe = std::move(result.recipe);
        selected_recipe = &compact_recipe.value();
    }
    if (selected_recipe == nullptr) {
        return std::nullopt;
    }

    auto materialized = legacy_parameters;
    materialized.program_config = selected_recipe->program_config;
    return materialized;
}

bool has_consistent_untilize_out(const Recipe& recipe) noexcept {
    if (const auto* config = std::get_if<MatmulMultiCoreReuseMultiCast1DProgramConfig>(&recipe.program_config)) {
        return config->untilize_out == recipe.untilize_out;
    }
    return !recipe.untilize_out;
}

std::optional<MatmulProgramConfig> materialize_registry_program_config(
    const compact::KeyDescriptor& key, const compact::ProgramConfigDescriptor& descriptor) {
    const compact::ProgramConfigCandidate candidate{.program_config = descriptor};
    if (!compact::legal_program_config_candidate(key, candidate)) {
        return std::nullopt;
    }
    const auto grid = tt::tt_metal::CoreCoord{descriptor.compute_grid_x, descriptor.compute_grid_y};
    // The fixed null/false/empty fields below are part of the versioned
    // acquisition policy validated by the lock emitter. out_block_* and the
    // MM1D receiver count are explicit because nanobind resolved their omitted
    // acquisition arguments to per-core M/N and one receiver respectively.
    switch (descriptor.family) {
        case compact::ProgramFamily::MultiCoreReuse:
            return MatmulProgramConfig{MatmulMultiCoreReuseProgramConfig{
                .compute_with_storage_grid_size = grid,
                .in0_block_w = descriptor.in0_block_w,
                .out_subblock_h = descriptor.out_subblock_h,
                .out_subblock_w = descriptor.out_subblock_w,
                .per_core_M = descriptor.per_core_m,
                .per_core_N = descriptor.per_core_n,
                .allowed_worker_cores = std::nullopt}};
        case compact::ProgramFamily::MultiCast1D:
            return MatmulProgramConfig{MatmulMultiCoreReuseMultiCast1DProgramConfig{
                .compute_with_storage_grid_size = grid,
                .in0_block_w = descriptor.in0_block_w,
                .out_subblock_h = descriptor.out_subblock_h,
                .out_subblock_w = descriptor.out_subblock_w,
                .out_block_h = descriptor.out_block_h,
                .out_block_w = descriptor.out_block_w,
                .per_core_M = descriptor.per_core_m,
                .per_core_N = descriptor.per_core_n,
                .fuse_batch = descriptor.fuse_batch,
                .fused_activation = std::nullopt,
                .mcast_in0 = descriptor.mcast_in0,
                .gather_in0 = false,
                .hop_cores = CoreRangeSet{},
                .num_global_cb_receivers = descriptor.num_global_cb_receivers,
                .untilize_out = false,
                .allowed_worker_cores = std::nullopt,
                .stream_in1 = false}};
        case compact::ProgramFamily::MultiCast2D:
            return MatmulProgramConfig{MatmulMultiCoreReuseMultiCastProgramConfig{
                .compute_with_storage_grid_size = grid,
                .in0_block_w = descriptor.in0_block_w,
                .out_subblock_h = descriptor.out_subblock_h,
                .out_subblock_w = descriptor.out_subblock_w,
                .out_block_h = descriptor.out_block_h,
                .out_block_w = descriptor.out_block_w,
                .per_core_M = descriptor.per_core_m,
                .per_core_N = descriptor.per_core_n,
                .transpose_mcast = descriptor.transpose_mcast,
                .fused_activation = std::nullopt,
                .fuse_batch = descriptor.fuse_batch,
                .allowed_worker_cores = std::nullopt}};
    }
    return std::nullopt;
}

MaterializationResult materialize_matmul_registry_recipe(const compact::EntryDescriptor& descriptor) {
    if (descriptor.key.schema_version != 1 || descriptor.key.codegen_recipe_abi != compact::kCodegenRecipeAbi ||
        descriptor.replay.schema_version != 2) {
        return {.status = MaterializationStatus::UnsupportedSchema};
    }
    if (descriptor.replay.family != compact::ProgramFamily::MultiCoreReuse) {
        return {.status = MaterializationStatus::UnsupportedReplay};
    }

    const auto& program = descriptor.replay.program_config;
    const auto maximum_subblock_area =
        descriptor.replay.compute_kernel_config.fp32_dest_acc_en ? std::uint32_t{4} : std::uint32_t{8};
    if (program.compute_grid_x == 0 || program.compute_grid_y == 0 || program.in0_block_w == 0 ||
        program.out_subblock_h == 0 || program.out_subblock_w == 0 || program.per_core_m == 0 ||
        program.per_core_n == 0 || program.allowed_worker_cores_present ||
        program.compute_grid_x > descriptor.key.compute_grid_x ||
        program.compute_grid_y > descriptor.key.compute_grid_y ||
        program.out_subblock_h > maximum_subblock_area / program.out_subblock_w ||
        !has_valid_multi_core_reuse_work_split(descriptor.key, program)) {
        return {.status = MaterializationStatus::InvalidProgramConfig};
    }

    const auto& state = descriptor.replay.call_state;
    const auto& key = descriptor.key;
    const bool scalar_semantics_valid = [&key] {
        switch (key.domain) {
            case compact::Domain::DenseMatmul:
            case compact::Domain::DenseLinear: return key.alpha_f32_bits == 0 && key.beta_f32_bits == 0;
            case compact::Domain::DenseAddmm:
                return key.alpha_f32_bits == 0x3F800000U &&
                       (key.beta_f32_bits == 0 || key.beta_f32_bits == 0x80000000U);
        }
        return false;
    }();
    const bool invalid_key_envelope =
        key.bcast_batch_present || key.bcast_batch || key.has_activation || key.has_bias || key.run_batched ||
        key.transpose_a || key.transpose_b || key.untilize_out || !scalar_semantics_valid ||
        key.input_a.layout != compact::Layout::Tile || key.input_b.layout != compact::Layout::Tile ||
        key.input_a.tile_height != 32 || key.input_a.tile_width != 32 || key.input_b.tile_height != 32 ||
        key.input_b.tile_width != 32 || key.output.buffer_type != compact::BufferType::Dram ||
        key.output.layout != compact::Layout::Tile || key.output.memory_layout != compact::MemoryLayout::Interleaved ||
        key.output.tile_height != 32 || key.output.tile_width != 32;
    if (invalid_key_envelope || state.output != key.output || state.untilize_out || !state.bcast_batch_is_null ||
        !state.user_core_coord_is_null || !state.user_fused_activation_is_null || !state.user_run_batched_is_false ||
        !state.transpose_a_is_false || !state.transpose_b_is_false || !state.output_tile_is_null ||
        !state.global_cb_is_null || !state.sub_device_id_is_null) {
        return {.status = MaterializationStatus::InvalidCallState};
    }

    tt::tt_metal::MathFidelity fidelity;
    switch (descriptor.replay.compute_kernel_config.math_fidelity) {
        case compact::MathFidelity::LoFi: fidelity = tt::tt_metal::MathFidelity::LoFi; break;
        case compact::MathFidelity::HiFi2: fidelity = tt::tt_metal::MathFidelity::HiFi2; break;
        case compact::MathFidelity::HiFi3: fidelity = tt::tt_metal::MathFidelity::HiFi3; break;
        case compact::MathFidelity::HiFi4: fidelity = tt::tt_metal::MathFidelity::HiFi4; break;
        default: return {.status = MaterializationStatus::InvalidComputeKernelConfig};
    }

    compute_throttle_utils::ThrottleLevel throttle;
    switch (descriptor.replay.compute_kernel_config.throttle_level) {
        case compact::ThrottleLevel::NoThrottle: throttle = compute_throttle_utils::ThrottleLevel::NO_THROTTLE; break;
        case compact::ThrottleLevel::Throttle1: throttle = compute_throttle_utils::ThrottleLevel::LEVEL_1; break;
        case compact::ThrottleLevel::Throttle2: throttle = compute_throttle_utils::ThrottleLevel::LEVEL_2; break;
        case compact::ThrottleLevel::Throttle3: throttle = compute_throttle_utils::ThrottleLevel::LEVEL_3; break;
        default: return {.status = MaterializationStatus::InvalidComputeKernelConfig};
    }

    const auto& kernel = descriptor.replay.compute_kernel_config;
    return MaterializationResult{
        .status = MaterializationStatus::Success,
        .recipe = Recipe{
            .program_config =
                MatmulMultiCoreReuseProgramConfig{
                    .compute_with_storage_grid_size =
                        tt::tt_metal::CoreCoord{program.compute_grid_x, program.compute_grid_y},
                    .in0_block_w = program.in0_block_w,
                    .out_subblock_h = program.out_subblock_h,
                    .out_subblock_w = program.out_subblock_w,
                    .per_core_M = program.per_core_m,
                    .per_core_N = program.per_core_n,
                    .allowed_worker_cores = std::nullopt},
            .compute_kernel_config =
                DeviceComputeKernelConfig{
                    .math_fidelity = fidelity,
                    .math_approx_mode = kernel.math_approx_mode,
                    .fp32_dest_acc_en = kernel.fp32_dest_acc_en,
                    .packer_l1_acc = kernel.packer_l1_acc,
                    .dst_full_sync_en = kernel.dst_full_sync_en,
                    .throttle_level = throttle},
            .untilize_out = state.untilize_out}};
}

ResolutionReason preflight_v1_eligibility(const Eligibility& eligibility) noexcept {
    if (eligibility.trace_capture_active) {
        return ResolutionReason::TraceCaptureUnsupported;
    }
    if (eligibility.call.domain == OperationDomain::IneligibleSharedCaller) {
        return ResolutionReason::IneligibleOperationDomain;
    }
    const bool is_addmm = eligibility.call.domain == OperationDomain::Addmm;
    const bool has_alpha = eligibility.call.alpha_f32_bits.has_value();
    const bool has_beta = eligibility.call.beta_f32_bits.has_value();
    if ((is_addmm && (!has_alpha || !has_beta)) || (!is_addmm && (has_alpha || has_beta))) {
        return ResolutionReason::MalformedOperationSemantics;
    }
    if (is_addmm && *eligibility.call.alpha_f32_bits != 0x3F800000U) {
        return ResolutionReason::MalformedOperationSemantics;
    }
    if (eligibility.io_contract_status != IoContractStatus::Resolved) {
        return ResolutionReason::InconsistentIoContract;
    }
    if (eligibility.has_program_config || eligibility.has_compute_kernel_config || eligibility.has_user_core_grid) {
        return ResolutionReason::ExplicitOverride;
    }
    // V1 deliberately admits the complete no-bias/no-activation/no-transpose
    // subset in every domain. Linear/addmm never alias dense.matmul, and their
    // richer tensor semantics remain ineligible until a later schema binds
    // every bias/activation field exactly.
    if (eligibility.has_bias || eligibility.has_activation || eligibility.transpose_a || eligibility.transpose_b ||
        eligibility.has_unsupported_tile_metadata || eligibility.has_optional_output || eligibility.has_output_tile ||
        eligibility.has_global_cb || eligibility.has_sub_device || eligibility.has_bcast_batch ||
        eligibility.untilize_out || eligibility.input_a_sharded || eligibility.input_b_sharded ||
        eligibility.output_sharded || eligibility.input_b_batched ||
        (is_addmm && *eligibility.call.beta_f32_bits != 0 && *eligibility.call.beta_f32_bits != 0x80000000U)) {
        return ResolutionReason::UnsupportedSemantics;
    }

    return ResolutionReason::CertifiedMatch;
}

ResolutionReason validate_v1_request_envelope(
    const MatmulRegistryRequest& request, const Eligibility& eligibility) noexcept {
    const auto preflight_reason = preflight_v1_eligibility(eligibility);
    if (preflight_reason != ResolutionReason::CertifiedMatch) {
        return preflight_reason;
    }
    if (request.schema_version != 1) {
        return ResolutionReason::IncompleteRequest;
    }
    const auto* const activation_end =
        request.activation_param_f32_bits.begin() +
        std::min<std::size_t>(request.activation_param_count, request.activation_param_f32_bits.size());
    const bool nonzero_activation_padding = std::any_of(
        activation_end, request.activation_param_f32_bits.end(), [](const auto value) { return value != 0; });
    if (request.call != eligibility.call || request.transpose_a != eligibility.transpose_a ||
        request.transpose_b != eligibility.transpose_b || request.has_bias != eligibility.has_bias ||
        request.has_activation != eligibility.has_activation || request.untilize_out != eligibility.untilize_out ||
        request.bcast_batch.has_value() != eligibility.has_bcast_batch ||
        request.run_batched != eligibility.input_b_batched ||
        request.has_activation != request.activation_op.has_value() ||
        request.activation_param_count > request.activation_param_f32_bits.size() ||
        (!request.has_activation && request.activation_param_count != 0) || nonzero_activation_padding) {
        return ResolutionReason::InconsistentRequest;
    }

    return ResolutionReason::CertifiedMatch;
}

static Resolution resolve_impl(
    const Mode mode,
    const MatmulRegistryRequest& request,
    const Eligibility& eligibility,
    const MatmulRegistryRequest* synthetic_request,
    const Recipe* synthetic_recipe,
    const compact::TableMetadata& metadata,
    const std::span<const compact::EntryDescriptor> entries,
    const CompatibilityDigests* actual_compatibility,
    const std::span<const compact::ProgramConfigGbdtModel> models,
    const std::span<const compact::ProgramConfigExactEntry> program_config_exact_entries) noexcept {
    if (mode == Mode::Off) {
        return {.reason = ResolutionReason::Disabled};
    }
    const auto envelope_reason = validate_v1_request_envelope(request, eligibility);
    if (envelope_reason != ResolutionReason::CertifiedMatch) {
        return {.reason = envelope_reason};
    }

    if (is_domain_circuit_broken(request.call.domain)) {
        return {.reason = ResolutionReason::CircuitBroken};
    }

    if (synthetic_request != nullptr && synthetic_recipe != nullptr && *synthetic_request == request) {
        return {.reason = ResolutionReason::CertifiedMatch, .recipe = synthetic_recipe};
    }

    std::size_t selectable_count = entries.size() + program_config_exact_entries.size();
    for (const auto& model : models) {
        selectable_count += model.candidates.size();
    }
    if (selectable_count == 0) {
        return {.reason = ResolutionReason::EmptyRegistry};
    }
    const bool direct_bank_scope = metadata.program_config_only_evidence_schema_version == 2 ||
                                   metadata.online_program_config_model_evidence_schema_version == 2;
    if (!direct_bank_scope && request.device.attestation_status != DeviceAttestationStatus::Success) {
        return {.reason = ResolutionReason::DeviceAttestationUnavailable};
    }
    const auto compatibility = actual_compatibility != nullptr
                                   ? validate_registry_compatibility(metadata, selectable_count, *actual_compatibility)
                                   : startup_compatibility_status();
    if (compatibility != CompatibilityStatus::Compatible) {
        return {.reason = compatibility_reason(compatibility)};
    }
    if (!direct_bank_scope && (metadata.runtime_capability_sha256 != request.device.runtime_capability_sha256 ||
                               (actual_compatibility != nullptr && actual_compatibility->runtime_capability_sha256 !=
                                                                       request.device.runtime_capability_sha256))) {
        return {.reason = ResolutionReason::RuntimeCapabilityMismatch};
    }

    const auto key = compact_registry_key(request);
    if (!key.has_value()) {
        return {.reason = ResolutionReason::IncompleteRequest};
    }
    if (metadata.program_config_only_evidence_schema_version != 0) {
        const auto* exact = direct_bank_scope
                                ? compact::lookup_program_config_exact_direct_bank(*key, program_config_exact_entries)
                                : compact::lookup_program_config_exact(*key, program_config_exact_entries);
        if (exact != nullptr) {
            if (!compact::legal_program_config_candidate(
                    *key,
                    compact::ProgramConfigCandidate{
                        .program_config = exact->program_config, .candidate_id = exact->entry_id})) {
                return {.reason = ResolutionReason::EmptyRegistry};
            }
            return {
                .reason = ResolutionReason::CertifiedMatch,
                .predicted_program_config = exact->program_config,
                .predicted_key = key,
            };
        }
        // Preserve the legacy table's unit-test ABI only when it is the sole
        // selectable representation. The production emitter never grants the
        // program-config-only evidence bit to a legacy-only lock. In a mixed
        // lock, a PC-only certificate must never authorize a distinct legacy
        // row after the typed exact table misses.
        if (program_config_exact_entries.empty() && models.empty()) {
            if (const auto* descriptor = compact::ExactIndex{entries}.lookup(*key); descriptor != nullptr) {
                return {
                    .reason = ResolutionReason::CertifiedMatch,
                    .descriptor = descriptor,
                    .predicted_program_config = compact::exact_program_config(descriptor->replay),
                    .predicted_key = key,
                };
            }
        }
    }
    // Legacy replay rows measured an explicit CKC. They remain readable for
    // lock compatibility but are never selectable, including in a mixed lock:
    // a PC-only exact certificate must not authorize a distinct legacy row.

    if (!models.empty() && metadata.online_program_config_model_evidence_schema_version == 0) {
        return {.reason = ResolutionReason::EmptyRegistry};
    }
    const compact::ProgramConfigGbdtModel* supported_model = nullptr;
    for (const auto& model : models) {
        if (compact::model_supports(*key, model, metadata.online_model_bundle_binding_sha256, direct_bank_scope)) {
            if (supported_model != nullptr) {
                // Overlapping support is invalid even if one model happens to
                // contain no legal candidates for this particular key.
                return {.reason = ResolutionReason::EmptyRegistry};
            }
            supported_model = &model;
        }
    }
    if (supported_model != nullptr) {
        const auto predicted_match = compact::lookup_program_config(
            *key,
            std::span<const compact::ProgramConfigExactEntry>{},
            *supported_model,
            metadata.online_model_bundle_binding_sha256,
            direct_bank_scope);
        if (predicted_match.source != compact::ProgramConfigLookupSource::Gbdt ||
            !predicted_match.program_config.has_value()) {
            return {.reason = ResolutionReason::EmptyRegistry};
        }
        return {
            .reason = ResolutionReason::PredictedMatch,
            .predicted_program_config = predicted_match.program_config,
            .predicted_key = key,
        };
    }

    return {.reason = ResolutionReason::EmptyRegistry};
}

Resolution resolve(const Mode mode, const MatmulRegistryRequest& request, const Eligibility& eligibility) noexcept {
    return resolve_impl(
        mode,
        request,
        eligibility,
        nullptr,
        nullptr,
        generated::metadata(),
        generated::entries(),
        nullptr,
        generated::online_models(),
        generated::program_config_exact_entries());
}

DispatchResult resolve_for_dispatch_decision(
    const Mode mode,
    const std::optional<MatmulRegistryRequest>& request,
    const Eligibility& eligibility,
    const ResolverFunction resolver) noexcept {
    auto resolution = Resolution{.reason = ResolutionReason::Disabled};
    if (mode != Mode::Off) {
        const auto preflight_reason = preflight_v1_eligibility(eligibility);
        if (preflight_reason != ResolutionReason::CertifiedMatch) {
            resolution = {.reason = preflight_reason};
        } else {
            resolution = request.has_value() && resolver != nullptr
                             ? resolver(mode, request.value(), eligibility)
                             : Resolution{.reason = ResolutionReason::IncompleteRequest};
        }
    }

    if (mode == Mode::On && resolution.reason == ResolutionReason::CertifiedMatch && resolution.recipe != nullptr &&
        !has_consistent_untilize_out(*resolution.recipe)) {
        circuit_break_domain(eligibility.call.domain);
        resolution.reason = ResolutionReason::MaterializationRejected;
    }
    return DispatchResult{.resolution = resolution, .action = execution_action(mode, resolution)};
}

DispatchResult resolve_for_dispatch(
    const Mode mode,
    const std::optional<MatmulRegistryRequest>& request,
    const Eligibility& eligibility,
    const ttnn::prim::MatmulParams& legacy_parameters,
    const ResolverFunction resolver,
    const MaterializerFunction materializer) {
    auto decision = resolve_for_dispatch_decision(mode, request, eligibility, resolver);
    auto& resolution = decision.resolution;
    auto& action = decision.action;
    std::optional<ttnn::prim::MatmulParams> materialized_parameters;
    try {
        if (action == ExecutionAction::ApplyRecipe && resolution.predicted_program_config.has_value() &&
            resolution.predicted_key.has_value()) {
            const auto program_config =
                materialize_registry_program_config(*resolution.predicted_key, *resolution.predicted_program_config);
            if (!program_config.has_value()) {
                resolution.reason = ResolutionReason::MaterializationRejected;
                action = ExecutionAction::Fallback;
                circuit_break_domain(eligibility.call.domain);
            } else {
                materialized_parameters = legacy_parameters;
                materialized_parameters->program_config = program_config;
            }
        } else if (action == ExecutionAction::ApplyRecipe && resolution.descriptor != nullptr) {
            auto materialized_recipe =
                materializer != nullptr ? materializer(*resolution.descriptor) : MaterializationResult{};
            if (materialized_recipe.status != MaterializationStatus::Success ||
                !materialized_recipe.recipe.has_value()) {
                const bool unsupported = materialized_recipe.status == MaterializationStatus::UnsupportedSchema ||
                                         materialized_recipe.status == MaterializationStatus::UnsupportedReplay;
                resolution.reason =
                    unsupported ? ResolutionReason::UnsupportedReplay : ResolutionReason::MaterializationRejected;
                action = ExecutionAction::Fallback;
                circuit_break_domain(eligibility.call.domain);
            } else {
                const auto native_resolution = Resolution{
                    .reason = ResolutionReason::CertifiedMatch, .recipe = &materialized_recipe.recipe.value()};
                materialized_parameters =
                    materialize_parameters_for_execution(mode, native_resolution, legacy_parameters);
            }
        } else {
            materialized_parameters = materialize_parameters_for_execution(mode, resolution, legacy_parameters);
        }
    } catch (...) {
        circuit_break_domain(eligibility.call.domain);
        resolution.reason = ResolutionReason::MaterializationRejected;
        action = ExecutionAction::Fallback;
        materialized_parameters.reset();
    }
    if (action == ExecutionAction::ApplyRecipe && !materialized_parameters.has_value()) {
        circuit_break_domain(eligibility.call.domain);
        resolution.reason = ResolutionReason::MaterializationRejected;
        action = ExecutionAction::Fallback;
    }
    return DispatchResult{
        .resolution = resolution, .action = action, .materialized_parameters = std::move(materialized_parameters)};
}

Resolution resolve_with_synthetic_candidate_for_testing(
    const Mode mode,
    const MatmulRegistryRequest& request,
    const Eligibility& eligibility,
    const MatmulRegistryRequest& candidate_request,
    const Recipe& candidate_recipe) noexcept {
    return resolve_impl(
        mode,
        request,
        eligibility,
        &candidate_request,
        &candidate_recipe,
        generated::metadata(),
        generated::entries(),
        nullptr,
        {},
        {});
}

Resolution resolve_with_compact_table_for_testing(
    const Mode mode,
    const MatmulRegistryRequest& request,
    const Eligibility& eligibility,
    const compact::TableMetadata& metadata,
    const std::span<const compact::EntryDescriptor> entries,
    const CompatibilityDigests& actual,
    const std::span<const compact::ProgramConfigGbdtModel> models,
    const std::span<const compact::ProgramConfigExactEntry> program_config_exact_entries) noexcept {
    return resolve_impl(
        mode, request, eligibility, nullptr, nullptr, metadata, entries, &actual, models, program_config_exact_entries);
}

bool circuit_break_domain(const OperationDomain domain) noexcept {
    if (index(domain) >= circuit_breakers.size()) {
        return false;
    }
    bool expected = false;
    if (!circuit_breakers[index(domain)].compare_exchange_strong(
            expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return false;
    }
    stats[index(domain)].circuit_breaker_activations.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool is_domain_circuit_broken(const OperationDomain domain) noexcept {
    return index(domain) < circuit_breakers.size() && circuit_breakers[index(domain)].load(std::memory_order_acquire);
}

void reset_circuit_breakers_for_testing() noexcept {
    for (auto& breaker : circuit_breakers) {
        breaker.store(false, std::memory_order_release);
    }
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
    if (resolution.reason == ResolutionReason::CertifiedMatch &&
        (resolution.recipe != nullptr || resolution.descriptor != nullptr ||
         (resolution.predicted_program_config.has_value() && resolution.predicted_key.has_value()))) {
        domain_stats.certified_hits.fetch_add(1, std::memory_order_relaxed);
    }
    switch (action) {
        case ExecutionAction::ObserveOnly:
            domain_stats.shadow_would_hits.fetch_add(1, std::memory_order_relaxed);
            break;
        case ExecutionAction::ApplyRecipe: domain_stats.selected_hits.fetch_add(1, std::memory_order_relaxed); break;
        case ExecutionAction::Fallback: domain_stats.fallbacks.fetch_add(1, std::memory_order_relaxed); break;
    }
}

void record_distributed_observation(const Mode mode, const DistributedMatmulClass classification) noexcept {
    if (mode == Mode::Off || index(classification) >= distributed_observations.size()) {
        return;
    }
    distributed_observations[index(classification)].fetch_add(1, std::memory_order_relaxed);
}

void record_completed_hit(const OperationDomain domain) noexcept {
    if (index(domain) < stats.size()) {
        stats[index(domain)].completed_hits.fetch_add(1, std::memory_order_relaxed);
    }
}

SelectedExecutionGuard::SelectedExecutionGuard(const OperationDomain domain, const bool* selected) noexcept :
    domain_(domain), selected_(selected), uncaught_exceptions_(std::uncaught_exceptions()) {}

SelectedExecutionGuard::~SelectedExecutionGuard() noexcept {
    if (selected_ == nullptr || !*selected_) {
        return;
    }
    if (std::uncaught_exceptions() > uncaught_exceptions_) {
        circuit_break_domain(domain_);
    } else {
        record_completed_hit(domain_);
    }
}

StatsSnapshot stats_snapshot() noexcept {
    StatsSnapshot snapshot;
    const auto mode = frozen_mode.load(std::memory_order_acquire);
    snapshot.mode_is_frozen = mode != kModeUninitialized;
    snapshot.frozen_mode = snapshot.mode_is_frozen ? static_cast<Mode>(mode) : Mode::Off;
    snapshot.compatibility_status = startup_compatibility_status();
    snapshot.table_metadata = generated::metadata();
    snapshot.entry_count = generated::entries().size();
    snapshot.entry_count += generated::program_config_exact_entries().size();
    for (const auto& model : generated::online_models()) {
        snapshot.entry_count += model.candidates.size();
    }

    for (std::size_t domain = 0; domain < stats.size(); ++domain) {
        const auto& source = stats[domain];
        auto& destination = snapshot.domains[domain];
        destination.resolution_attempts = source.resolution_attempts.load(std::memory_order_relaxed);
        destination.certified_hits = source.certified_hits.load(std::memory_order_relaxed);
        destination.shadow_would_hits = source.shadow_would_hits.load(std::memory_order_relaxed);
        destination.selected_hits = source.selected_hits.load(std::memory_order_relaxed);
        destination.completed_hits = source.completed_hits.load(std::memory_order_relaxed);
        destination.fallbacks = source.fallbacks.load(std::memory_order_relaxed);
        destination.circuit_breaker_activations = source.circuit_breaker_activations.load(std::memory_order_relaxed);
        destination.circuit_broken = circuit_breakers[domain].load(std::memory_order_acquire);
        for (std::size_t reason = 0; reason < kResolutionReasonCount; ++reason) {
            destination.reasons[reason] = source.reasons[reason].load(std::memory_order_relaxed);
        }
    }
    for (std::size_t classification = 0; classification < distributed_observations.size(); ++classification) {
        snapshot.distributed_observations[classification] =
            distributed_observations[classification].load(std::memory_order_relaxed);
    }
    return snapshot;
}

void reset_stats_for_testing() noexcept {
    for (auto& domain : stats) {
        domain.resolution_attempts.store(0, std::memory_order_relaxed);
        domain.certified_hits.store(0, std::memory_order_relaxed);
        domain.shadow_would_hits.store(0, std::memory_order_relaxed);
        domain.selected_hits.store(0, std::memory_order_relaxed);
        domain.completed_hits.store(0, std::memory_order_relaxed);
        domain.fallbacks.store(0, std::memory_order_relaxed);
        domain.circuit_breaker_activations.store(0, std::memory_order_relaxed);
        for (auto& reason : domain.reasons) {
            reason.store(0, std::memory_order_relaxed);
        }
    }
    for (auto& observation : distributed_observations) {
        observation.store(0, std::memory_order_relaxed);
    }
}

}  // namespace ttnn::operations::matmul::registry
