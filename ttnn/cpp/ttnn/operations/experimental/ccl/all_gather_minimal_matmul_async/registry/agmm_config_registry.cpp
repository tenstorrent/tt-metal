// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "agmm_config_registry.hpp"

#include <atomic>
#include <limits>

#include "agmm_registry_data.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"

namespace ttnn::experimental::all_gather_minimal_matmul_registry {
namespace {

constexpr std::uint8_t kModeUninitialized = 0xFF;

struct AtomicStats {
    std::atomic<std::uint64_t> resolution_attempts{0};
    std::atomic<std::uint64_t> certified_hits{0};
    std::atomic<std::uint64_t> shadow_would_hits{0};
    std::atomic<std::uint64_t> selected_hits{0};
    std::atomic<std::uint64_t> launch_completed_hits{0};
    std::atomic<std::uint64_t> fallbacks{0};
    std::atomic<std::uint64_t> circuit_breaker_activations{0};
    std::array<std::atomic<std::uint64_t>, kResolutionReasonCount> reasons{};
};

std::atomic<std::uint8_t> frozen_mode{kModeUninitialized};
std::atomic<bool> circuit_broken{false};
AtomicStats stats;

constexpr std::size_t index(const ResolutionReason reason) noexcept { return static_cast<std::size_t>(reason); }

bool is_zero(const compact::Sha256& digest) noexcept {
    for (const auto byte : digest) {
        if (byte != 0) {
            return false;
        }
    }
    return true;
}

bool is_default_tile(const compact::TensorDescriptor& tensor) noexcept {
    constexpr std::uint32_t kTileLayout = 1;
    constexpr std::uint16_t kTileExtent = 32;
    return tensor.layout == kTileLayout && tensor.tile_height == kTileExtent && tensor.tile_width == kTileExtent &&
           !tensor.tile_transpose_of_faces && !tensor.tile_transpose_within_face;
}

bool tensor_contract_is_complete(const compact::TensorDescriptor& tensor) noexcept {
    if (tensor.rank == 0 || tensor.rank > compact::kMaxTensorRank || is_zero(tensor.memory_config_sha256) ||
        is_zero(tensor.tensor_topology_sha256)) {
        return false;
    }
    for (std::size_t axis = 0; axis < compact::kMaxTensorRank; ++axis) {
        if (axis < tensor.rank) {
            if (tensor.logical_shape[axis] == 0 || tensor.padded_shape[axis] < tensor.logical_shape[axis]) {
                return false;
            }
        } else if (tensor.logical_shape[axis] != 0 || tensor.padded_shape[axis] != 0) {
            return false;
        }
    }
    return true;
}

bool optional_tensor_contract_is_complete(const compact::OptionalTensorDescriptor& tensor) noexcept {
    if (!tensor.present) {
        return tensor.tensor == compact::TensorDescriptor{};
    }
    return tensor_contract_is_complete(tensor.tensor) && is_default_tile(tensor.tensor);
}

bool operation_contract_is_complete(const compact::KeyDescriptor& key) noexcept;
bool shape_matches_workload(const compact::KeyDescriptor& key) noexcept;

bool device_contract_has_required_facts(const compact::DeviceDescriptor& device) noexcept {
    return device.architecture == compact::kBlackholeArchitecture && device.board_capability_class != 0 &&
           device.device_count == compact::kBh32DeviceCount && device.mesh_rows == compact::kBh32MeshRows &&
           device.mesh_cols == compact::kBh32MeshCols && device.compute_grid_x != 0 && device.compute_grid_y != 0 &&
           !is_zero(device.ordered_mesh_sha256) && !is_zero(device.topology_sha256) &&
           !is_zero(device.runtime_capability_sha256);
}

bool request_has_required_facts(const compact::KeyDescriptor& key) noexcept {
    const auto optional_has_required_facts = [](const compact::OptionalTensorDescriptor& tensor) {
        return !tensor.present || tensor_contract_is_complete(tensor.tensor);
    };
    return key.schema_version == compact::kKeySchemaVersion && key.codegen_recipe_abi == compact::kCodegenRecipeAbi &&
           device_contract_has_required_facts(key.device) && key.workload.logical_m != 0 &&
           key.workload.logical_k != 0 && key.workload.logical_n != 0 && key.workload.padded_m != 0 &&
           key.workload.padded_k != 0 && key.workload.padded_n != 0 && key.workload.batch != 0 &&
           tensor_contract_is_complete(key.input) && tensor_contract_is_complete(key.weight) &&
           optional_has_required_facts(key.bias) && optional_has_required_facts(key.ternary_input_a) &&
           optional_has_required_facts(key.ternary_input_b) && optional_has_required_facts(key.persistent_output) &&
           optional_has_required_facts(key.persistent_weight) && key.operation.ring_size != 0 &&
           key.operation.fsdp_ring_size != 0 && key.operation.output_tile_height != 0 &&
           key.operation.output_tile_width != 0 && !is_zero(key.operation.output_memory_config_sha256) &&
           !is_zero(key.operation.output_tensor_topology_sha256);
}

bool request_contract_is_consistent(const compact::KeyDescriptor& key) noexcept {
    return static_cast<std::uint32_t>(key.device.mesh_rows) * key.device.mesh_cols == key.device.device_count &&
           is_default_tile(key.input) && is_default_tile(key.weight) &&
           optional_tensor_contract_is_complete(key.bias) &&
           optional_tensor_contract_is_complete(key.ternary_input_a) &&
           optional_tensor_contract_is_complete(key.ternary_input_b) &&
           optional_tensor_contract_is_complete(key.persistent_output) &&
           optional_tensor_contract_is_complete(key.persistent_weight) && operation_contract_is_complete(key) &&
           key.operation.output_layout == 1 && key.operation.output_tile_height == 32 &&
           key.operation.output_tile_width == 32 && !key.operation.output_tile_transpose_of_faces &&
           !key.operation.output_tile_transpose_within_face && shape_matches_workload(key);
}

bool operation_contract_is_complete(const compact::KeyDescriptor& key) noexcept {
    const auto& operation = key.operation;
    if (operation.chunks < 1 || operation.dim != -1 || operation.chunk_size_count > compact::kMaxChunkSizes ||
        operation.activation_parameter_count > compact::kMaxActivationParameters ||
        is_zero(operation.output_memory_config_sha256) || is_zero(operation.output_tensor_topology_sha256)) {
        return false;
    }
    if ((!operation.scalar_present && operation.scalar_f32_bits != 0) ||
        (operation.scalar_present != (key.ternary_input_a.present && key.ternary_input_b.present)) ||
        (key.ternary_input_a.present != key.ternary_input_b.present) ||
        (operation.persistent_output_present != key.persistent_output.present) ||
        (operation.persistent_weight_present != key.persistent_weight.present) ||
        (operation.scalar_present && operation.activation_present) ||
        (operation.fuse_swiglu &&
         (operation.scalar_present || operation.activation_present || operation.chunks != 1))) {
        return false;
    }
    if ((!operation.cluster_axis_present && operation.cluster_axis != 0) ||
        (!operation.fsdp_cluster_axis_present && operation.fsdp_cluster_axis != 0) ||
        (!operation.activation_present &&
         (operation.activation_op != 0 || operation.activation_parameter_count != 0))) {
        return false;
    }
    constexpr std::uint32_t kLinearTopology = 1;
    if (operation.fsdp_cluster_axis_present) {
        if (operation.fsdp_ring_size <= 1 || operation.ring_size != operation.fsdp_ring_size ||
            operation.topology != kLinearTopology || operation.fsdp_topology != kLinearTopology ||
            !operation.persistent_weight_present || operation.fsdp_semaphore_count < 2 ||
            (operation.cluster_axis_present && operation.cluster_axis == operation.fsdp_cluster_axis)) {
            return false;
        }
    } else if (operation.fsdp_ring_size != 1) {
        return false;
    }
    for (std::size_t index = operation.activation_parameter_count; index < compact::kMaxActivationParameters; ++index) {
        if (operation.activation_parameter_f32_bits[index] != 0) {
            return false;
        }
    }
    std::uint64_t chunk_sum = 0;
    for (std::size_t index = 0; index < compact::kMaxChunkSizes; ++index) {
        const auto width = operation.chunk_sizes[index];
        if (index < operation.chunk_size_count) {
            if (width == 0 || width % 32 != 0) {
                return false;
            }
            chunk_sum += width;
        } else if (width != 0) {
            return false;
        }
    }
    if (operation.chunk_size_count != 0) {
        if (operation.chunk_size_count != static_cast<std::size_t>(operation.chunks) ||
            chunk_sum != key.workload.logical_n) {
            return false;
        }
    } else if (
        operation.chunks > 1 &&
        (key.workload.logical_n % operation.chunks != 0 || (key.workload.logical_n / operation.chunks) % 32 != 0)) {
        return false;
    }
    return true;
}

bool shape_matches_workload(const compact::KeyDescriptor& key) noexcept {
    const auto& input = key.input;
    const auto& weight = key.weight;
    if (input.rank < 2 || input.rank > compact::kMaxTensorRank || weight.rank < 2 ||
        weight.rank > compact::kMaxTensorRank || key.operation.ring_size == 0 || key.operation.fsdp_ring_size == 0) {
        return false;
    }
    const auto input_m = static_cast<std::size_t>(input.rank - 2);
    const auto input_k = static_cast<std::size_t>(input.rank - 1);
    const auto weight_k = static_cast<std::size_t>(weight.rank - 2);
    const auto weight_n = static_cast<std::size_t>(weight.rank - 1);
    const auto checked_product = [](const std::uint64_t lhs, const std::uint64_t rhs) -> std::optional<std::uint64_t> {
        if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
            return std::nullopt;
        }
        return lhs * rhs;
    };
    const auto logical_k = checked_product(input.logical_shape[input_k], key.operation.ring_size);
    const auto logical_weight_k = checked_product(weight.logical_shape[weight_k], key.operation.fsdp_ring_size);
    const auto padded_k = checked_product(input.padded_shape[input_k], key.operation.ring_size);
    const auto padded_weight_k = checked_product(weight.padded_shape[weight_k], key.operation.fsdp_ring_size);
    if (!logical_k.has_value() || !logical_weight_k.has_value() || !padded_k.has_value() ||
        !padded_weight_k.has_value() || logical_k != logical_weight_k || padded_k != padded_weight_k) {
        return false;
    }
    std::uint64_t batch = 1;
    for (std::size_t axis = 0; axis < input_m; ++axis) {
        const auto next = checked_product(batch, input.logical_shape[axis]);
        if (!next.has_value()) {
            return false;
        }
        batch = *next;
    }
    return key.workload.logical_m == input.logical_shape[input_m] && key.workload.logical_k == *logical_k &&
           key.workload.logical_n == weight.logical_shape[weight_n] &&
           key.workload.padded_m == input.padded_shape[input_m] && key.workload.padded_k == *padded_k &&
           key.workload.padded_n == weight.padded_shape[weight_n] && key.workload.batch == batch;
}

ResolutionReason compatibility_reason(const CompatibilityStatus status) noexcept {
    switch (status) {
        case CompatibilityStatus::Compatible: return ResolutionReason::CertifiedMatch;
        case CompatibilityStatus::EmptyRegistry: return ResolutionReason::EmptyRegistry;
        case CompatibilityStatus::Unavailable: return ResolutionReason::CompatibilityUnavailable;
        case CompatibilityStatus::SchemaMismatch:
        case CompatibilityStatus::MalformedTable:
        case CompatibilityStatus::DeviceMismatch:
        case CompatibilityStatus::DigestMismatch: return ResolutionReason::CompatibilityMismatch;
    }
    return ResolutionReason::CompatibilityUnavailable;
}

Resolution resolve_impl(
    const Mode mode,
    const std::optional<RegistryRequest>& request,
    const AttestationStatus attestation_status,
    const Eligibility& eligibility,
    const compact::TableLock& lock,
    const std::span<const compact::EntryDescriptor> entries,
    const CompatibilityDigests& actual) noexcept {
    if (mode == Mode::Off) {
        return {.reason = ResolutionReason::Disabled};
    }
    if (const auto reason = preflight(eligibility); reason != ResolutionReason::CertifiedMatch) {
        return {.reason = reason};
    }
    if (is_circuit_broken()) {
        return {.reason = ResolutionReason::CircuitBroken};
    }
    // Empty production storage is checked before request/attestation work so
    // empty Shadow and On remain cheap and behavior-preserving.
    if (entries.empty()) {
        return {.reason = ResolutionReason::EmptyRegistry};
    }
    if (attestation_status == AttestationStatus::UnsupportedAttestation) {
        return {.reason = ResolutionReason::UnsupportedAttestation};
    }
    if (attestation_status == AttestationStatus::QueryFailed) {
        return {.reason = ResolutionReason::AttestationQueryFailed};
    }
    if (!request.has_value()) {
        return {.reason = ResolutionReason::IncompleteRequest};
    }
    const auto compatibility = validate_compatibility(lock, entries, actual, request->key.device);
    if (compatibility != CompatibilityStatus::Compatible) {
        return {.reason = compatibility_reason(compatibility)};
    }
    if (request->key.schema_version != compact::kKeySchemaVersion ||
        request->key.codegen_recipe_abi != compact::kCodegenRecipeAbi) {
        return {.reason = ResolutionReason::IncompleteRequest};
    }
    if (const auto* descriptor = compact::lookup_exact(request->key, entries); descriptor != nullptr) {
        return {.reason = ResolutionReason::CertifiedMatch, .descriptor = descriptor};
    }
    return {.reason = ResolutionReason::ExactMiss};
}

}  // namespace

std::string_view resolution_reason_name(const ResolutionReason reason) noexcept {
    switch (reason) {
        case ResolutionReason::Disabled: return "disabled";
        case ResolutionReason::TraceCaptureUnsupported: return "trace_capture_unsupported";
        case ResolutionReason::ExplicitProgramConfig: return "explicit_program_config";
        case ResolutionReason::ExplicitComputeKernelConfig: return "explicit_compute_kernel_config";
        case ResolutionReason::UnsupportedAttestation: return "unsupported_attestation";
        case ResolutionReason::AttestationQueryFailed: return "attestation_query_failed";
        case ResolutionReason::IncompleteRequest: return "incomplete_request";
        case ResolutionReason::CompatibilityUnavailable: return "compatibility_unavailable";
        case ResolutionReason::CompatibilityMismatch: return "compatibility_mismatch";
        case ResolutionReason::CircuitBroken: return "circuit_broken";
        case ResolutionReason::EmptyRegistry: return "empty_registry";
        case ResolutionReason::ExactMiss: return "exact_miss";
        case ResolutionReason::UnsupportedReplay: return "unsupported_replay";
        case ResolutionReason::MaterializationRejected: return "materialization_rejected";
        case ResolutionReason::CertifiedMatch: return "certified_match";
        case ResolutionReason::Count: return "count";
    }
    return "unknown";
}

AttestationResult production_attestation(const tt::tt_metal::distributed::MeshDevice&) noexcept {
    return {.status = AttestationStatus::UnsupportedAttestation};
}

RequestBuildResult build_registry_request(const RegistryRequestFacts& facts) noexcept {
    switch (preflight(facts.eligibility)) {
        case ResolutionReason::TraceCaptureUnsupported:
            return {.status = RequestBuildStatus::TraceCaptureUnsupported};
        case ResolutionReason::ExplicitProgramConfig:
            return {.status = RequestBuildStatus::ExplicitProgramConfig};
        case ResolutionReason::ExplicitComputeKernelConfig:
            return {.status = RequestBuildStatus::ExplicitComputeKernelConfig};
        case ResolutionReason::CertifiedMatch: break;
        default: return {.status = RequestBuildStatus::IncompleteDescriptor};
    }
    if (facts.attestation.status == AttestationStatus::UnsupportedAttestation) {
        return {.status = RequestBuildStatus::UnsupportedAttestation};
    }
    if (facts.attestation.status == AttestationStatus::QueryFailed) {
        return {.status = RequestBuildStatus::AttestationQueryFailed};
    }

    const auto key = compact::KeyDescriptor{
        .device = facts.attestation.device,
        .workload = facts.workload,
        .operation = facts.operation,
        .input = facts.input,
        .weight = facts.weight,
        .bias = facts.bias,
        .ternary_input_a = facts.ternary_input_a,
        .ternary_input_b = facts.ternary_input_b,
        .persistent_output = facts.persistent_output,
        .persistent_weight = facts.persistent_weight};
    if (!request_has_required_facts(key)) {
        return {.status = RequestBuildStatus::IncompleteDescriptor};
    }
    if (!request_contract_is_consistent(key)) {
        return {.status = RequestBuildStatus::InconsistentDescriptor};
    }
    return {.status = RequestBuildStatus::Success, .request = RegistryRequest{.key = key}};
}

CompatibilityStatus validate_compatibility(
    const compact::TableLock& lock,
    const std::span<const compact::EntryDescriptor> entries,
    const CompatibilityDigests& actual,
    const compact::DeviceDescriptor& device) noexcept {
    switch (compact::validate_table_lock(lock, entries)) {
        case compact::TableValidationStatus::Valid: break;
        case compact::TableValidationStatus::Empty: return CompatibilityStatus::EmptyRegistry;
        case compact::TableValidationStatus::LockSchemaMismatch: return CompatibilityStatus::SchemaMismatch;
        case compact::TableValidationStatus::EntryCountMismatch:
        case compact::TableValidationStatus::MissingLockDigest:
        case compact::TableValidationStatus::EntrySchemaMismatch:
        case compact::TableValidationStatus::MissingEntryId:
        case compact::TableValidationStatus::UnsupportedDeviceDomain:
        case compact::TableValidationStatus::CertifiedDeviceMismatch:
        case compact::TableValidationStatus::EntriesNotStrictlySorted:
            return CompatibilityStatus::MalformedTable;
    }
    if (is_zero(actual.semantic_source_sha256) || is_zero(actual.build_identity_sha256) ||
        is_zero(actual.runtime_capability_sha256)) {
        return CompatibilityStatus::Unavailable;
    }
    const auto& metadata = lock.metadata;
    if (device != lock.certified_device) {
        return CompatibilityStatus::DeviceMismatch;
    }
    if (metadata.semantic_source_sha256 != actual.semantic_source_sha256 ||
        metadata.build_identity_sha256 != actual.build_identity_sha256 ||
        metadata.runtime_capability_sha256 != actual.runtime_capability_sha256 ||
        actual.runtime_capability_sha256 != device.runtime_capability_sha256) {
        return CompatibilityStatus::DigestMismatch;
    }
    return CompatibilityStatus::Compatible;
}

MaterializationResult materialize_recipe(const compact::EntryDescriptor& descriptor) noexcept {
    if (descriptor.key.schema_version != compact::kKeySchemaVersion ||
        descriptor.key.codegen_recipe_abi != compact::kCodegenRecipeAbi ||
        descriptor.replay.schema_version != compact::kReplaySchemaVersion) {
        return {.status = MaterializationStatus::UnsupportedSchema};
    }

    if (!device_contract_has_required_facts(descriptor.key.device) ||
        !tensor_contract_is_complete(descriptor.key.input) || !tensor_contract_is_complete(descriptor.key.weight) ||
        !is_default_tile(descriptor.key.input) || !is_default_tile(descriptor.key.weight) ||
        !optional_tensor_contract_is_complete(descriptor.key.bias) ||
        !optional_tensor_contract_is_complete(descriptor.key.ternary_input_a) ||
        !optional_tensor_contract_is_complete(descriptor.key.ternary_input_b) ||
        !optional_tensor_contract_is_complete(descriptor.key.persistent_output) ||
        !optional_tensor_contract_is_complete(descriptor.key.persistent_weight) ||
        !operation_contract_is_complete(descriptor.key) || descriptor.key.operation.output_layout != 1 ||
        descriptor.key.operation.output_tile_height != 32 || descriptor.key.operation.output_tile_width != 32 ||
        descriptor.key.operation.output_tile_transpose_of_faces ||
        descriptor.key.operation.output_tile_transpose_within_face || !shape_matches_workload(descriptor.key)) {
        return {.status = MaterializationStatus::InvalidProgramConfig};
    }

    const auto& config = descriptor.replay.config;
    if (config.m_block_size == 0 || config.k_block_size == 0 || config.n_block_size == 0 || config.subblock_h == 0 ||
        config.subblock_w == 0 || config.compute_grid_x == 0 || config.compute_grid_y == 0 ||
        config.compute_grid_x < 2 || config.compute_grid_y < 2 ||
        (descriptor.key.operation.fuse_swiglu && config.n_block_size % 2 != 0) ||
        config.m_block_size % config.subblock_h != 0 || config.n_block_size % config.subblock_w != 0 ||
        config.compute_grid_x > descriptor.key.device.compute_grid_x ||
        config.compute_grid_y > descriptor.key.device.compute_grid_y) {
        return {.status = MaterializationStatus::InvalidProgramConfig};
    }
    constexpr std::uint64_t kTileExtent = 32;
    const auto local_k_tiles = descriptor.key.input.padded_shape[descriptor.key.input.rank - 1] / kTileExtent;
    constexpr std::uint32_t kLinearTopology = 1;
    if (config.k_block_size > local_k_tiles ||
        (descriptor.key.operation.topology != kLinearTopology && local_k_tiles % config.k_block_size != 0)) {
        return {.status = MaterializationStatus::InvalidProgramConfig};
    }

    tt::tt_metal::MathFidelity fidelity;
    switch (descriptor.replay.compute_kernel_config.math_fidelity) {
        case static_cast<std::uint32_t>(tt::tt_metal::MathFidelity::LoFi):
            fidelity = tt::tt_metal::MathFidelity::LoFi;
            break;
        case static_cast<std::uint32_t>(tt::tt_metal::MathFidelity::HiFi2):
            fidelity = tt::tt_metal::MathFidelity::HiFi2;
            break;
        case static_cast<std::uint32_t>(tt::tt_metal::MathFidelity::HiFi3):
            fidelity = tt::tt_metal::MathFidelity::HiFi3;
            break;
        case static_cast<std::uint32_t>(tt::tt_metal::MathFidelity::HiFi4):
            fidelity = tt::tt_metal::MathFidelity::HiFi4;
            break;
        default: return {.status = MaterializationStatus::InvalidComputeKernelConfig};
    }

    using ThrottleLevel = ttnn::operations::compute_throttle_utils::ThrottleLevel;
    ThrottleLevel throttle;
    switch (descriptor.replay.compute_kernel_config.throttle_level) {
        case static_cast<std::uint32_t>(ThrottleLevel::NO_THROTTLE): throttle = ThrottleLevel::NO_THROTTLE; break;
        case static_cast<std::uint32_t>(ThrottleLevel::LEVEL_1): throttle = ThrottleLevel::LEVEL_1; break;
        case static_cast<std::uint32_t>(ThrottleLevel::LEVEL_2): throttle = ThrottleLevel::LEVEL_2; break;
        case static_cast<std::uint32_t>(ThrottleLevel::LEVEL_3): throttle = ThrottleLevel::LEVEL_3; break;
        case static_cast<std::uint32_t>(ThrottleLevel::LEVEL_4): throttle = ThrottleLevel::LEVEL_4; break;
        case static_cast<std::uint32_t>(ThrottleLevel::LEVEL_5): throttle = ThrottleLevel::LEVEL_5; break;
        default: return {.status = MaterializationStatus::InvalidComputeKernelConfig};
    }

    const auto& kernel = descriptor.replay.compute_kernel_config;
    const auto materialized_kernel = DeviceComputeKernelConfig{
        .math_fidelity = fidelity,
        .math_approx_mode = kernel.math_approx_mode,
        .fp32_dest_acc_en = kernel.fp32_dest_acc_en,
        .packer_l1_acc = kernel.packer_l1_acc,
        .dst_full_sync_en = kernel.dst_full_sync_en,
        .throttle_level = throttle};
    const auto maximum_subblock_area = get_dest_reg_count(materialized_kernel);
    if (config.subblock_h > maximum_subblock_area / config.subblock_w) {
        return {.status = MaterializationStatus::InvalidProgramConfig};
    }
    return MaterializationResult{
        .status = MaterializationStatus::Success,
        .recipe = Recipe{
            .config =
                ttnn::experimental::prim::MinimalMatmulConfig{
                    .M_block_size = config.m_block_size,
                    .K_block_size = config.k_block_size,
                    .N_block_size = config.n_block_size,
                    .subblock_h = config.subblock_h,
                    .subblock_w = config.subblock_w,
                    .compute_with_storage_grid_size = {config.compute_grid_x, config.compute_grid_y}},
            .compute_kernel_config = materialized_kernel}};
}

ResolutionReason preflight(const Eligibility& eligibility) noexcept {
    if (eligibility.trace_capture_active) {
        return ResolutionReason::TraceCaptureUnsupported;
    }
    if (eligibility.has_explicit_program_config) {
        return ResolutionReason::ExplicitProgramConfig;
    }
    if (eligibility.has_explicit_compute_kernel_config) {
        return ResolutionReason::ExplicitComputeKernelConfig;
    }
    return ResolutionReason::CertifiedMatch;
}

Resolution resolve(
    const Mode mode,
    const std::optional<RegistryRequest>& request,
    const AttestationStatus attestation_status,
    const Eligibility& eligibility) noexcept {
    // No production entry may become active until independently generated
    // compatibility digests and exact multi-device attestation are wired.
    return resolve_impl(
        mode,
        request,
        attestation_status,
        eligibility,
        generated::lock(),
        generated::entries(),
        CompatibilityDigests{});
}

DispatchResult resolve_for_dispatch(
    const Mode mode,
    const std::optional<RegistryRequest>& request,
    const AttestationStatus attestation_status,
    const Eligibility& eligibility,
    const ResolverFunction resolver,
    const MaterializerFunction materializer) noexcept {
    auto resolution = Resolution{.reason = ResolutionReason::Disabled};
    if (mode != Mode::Off) {
        resolution = resolver != nullptr ? resolver(mode, request, attestation_status, eligibility)
                                         : Resolution{.reason = ResolutionReason::IncompleteRequest};
    }

    auto action = ExecutionAction::Fallback;
    std::optional<Recipe> recipe;
    if (resolution.reason == ResolutionReason::CertifiedMatch && resolution.descriptor != nullptr) {
        if (mode == Mode::Shadow) {
            action = ExecutionAction::ObserveOnly;
        } else if (mode == Mode::On) {
            try {
                const auto materialized =
                    materializer != nullptr ? materializer(*resolution.descriptor) : MaterializationResult{};
                if (materialized.status == MaterializationStatus::Success && materialized.recipe.has_value()) {
                    recipe = materialized.recipe;
                    action = ExecutionAction::ApplyRecipe;
                } else {
                    resolution.reason = materialized.status == MaterializationStatus::UnsupportedSchema
                                            ? ResolutionReason::UnsupportedReplay
                                            : ResolutionReason::MaterializationRejected;
                    circuit_break();
                }
            } catch (...) {
                resolution.reason = ResolutionReason::MaterializationRejected;
                circuit_break();
            }
        }
    }
    return DispatchResult{.resolution = resolution, .action = action, .recipe = recipe};
}

Resolution resolve_with_table_for_testing(
    const Mode mode,
    const std::optional<RegistryRequest>& request,
    const AttestationStatus attestation_status,
    const Eligibility& eligibility,
    const compact::TableLock& lock,
    const std::span<const compact::EntryDescriptor> entries,
    const CompatibilityDigests& actual) noexcept {
    return resolve_impl(mode, request, attestation_status, eligibility, lock, entries, actual);
}

Mode current_mode() noexcept {
    auto value = frozen_mode.load(std::memory_order_acquire);
    if (value != kModeUninitialized) {
        return static_cast<Mode>(value);
    }
    const auto configured = ttnn::CONFIG.get<"agmm_registry_mode">();
    const auto raw = static_cast<std::uint8_t>(configured);
    const auto bounded = raw <= static_cast<std::uint8_t>(Mode::On) ? raw : static_cast<std::uint8_t>(Mode::Off);
    if (frozen_mode.compare_exchange_strong(value, bounded, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return static_cast<Mode>(bounded);
    }
    return static_cast<Mode>(value);
}

void reset_startup_mode_for_testing() noexcept { frozen_mode.store(kModeUninitialized, std::memory_order_release); }

bool circuit_break() noexcept {
    bool expected = false;
    if (!circuit_broken.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return false;
    }
    stats.circuit_breaker_activations.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool is_circuit_broken() noexcept { return circuit_broken.load(std::memory_order_acquire); }

void reset_circuit_breaker_for_testing() noexcept { circuit_broken.store(false, std::memory_order_release); }

void record_resolution(const Mode mode, const Resolution& resolution, const ExecutionAction action) noexcept {
    if (mode == Mode::Off || index(resolution.reason) >= kResolutionReasonCount) {
        return;
    }
    stats.resolution_attempts.fetch_add(1, std::memory_order_relaxed);
    stats.reasons[index(resolution.reason)].fetch_add(1, std::memory_order_relaxed);
    if (resolution.reason == ResolutionReason::CertifiedMatch && resolution.descriptor != nullptr) {
        stats.certified_hits.fetch_add(1, std::memory_order_relaxed);
    }
    switch (action) {
        case ExecutionAction::ObserveOnly: stats.shadow_would_hits.fetch_add(1, std::memory_order_relaxed); break;
        case ExecutionAction::ApplyRecipe: stats.selected_hits.fetch_add(1, std::memory_order_relaxed); break;
        case ExecutionAction::Fallback: stats.fallbacks.fetch_add(1, std::memory_order_relaxed); break;
    }
}

void record_launch_completed_hit() noexcept { stats.launch_completed_hits.fetch_add(1, std::memory_order_relaxed); }

StatsSnapshot stats_snapshot() noexcept {
    StatsSnapshot snapshot;
    const auto mode = frozen_mode.load(std::memory_order_acquire);
    snapshot.mode_is_frozen = mode != kModeUninitialized;
    snapshot.frozen_mode = snapshot.mode_is_frozen ? static_cast<Mode>(mode) : Mode::Off;
    snapshot.entry_count = generated::entries().size();
    snapshot.resolution_attempts = stats.resolution_attempts.load(std::memory_order_relaxed);
    snapshot.certified_hits = stats.certified_hits.load(std::memory_order_relaxed);
    snapshot.shadow_would_hits = stats.shadow_would_hits.load(std::memory_order_relaxed);
    snapshot.selected_hits = stats.selected_hits.load(std::memory_order_relaxed);
    snapshot.launch_completed_hits = stats.launch_completed_hits.load(std::memory_order_relaxed);
    snapshot.fallbacks = stats.fallbacks.load(std::memory_order_relaxed);
    snapshot.circuit_breaker_activations = stats.circuit_breaker_activations.load(std::memory_order_relaxed);
    snapshot.circuit_broken = is_circuit_broken();
    for (std::size_t reason = 0; reason < kResolutionReasonCount; ++reason) {
        snapshot.reasons[reason] = stats.reasons[reason].load(std::memory_order_relaxed);
    }
    return snapshot;
}

void reset_stats_for_testing() noexcept {
    stats.resolution_attempts.store(0, std::memory_order_relaxed);
    stats.certified_hits.store(0, std::memory_order_relaxed);
    stats.shadow_would_hits.store(0, std::memory_order_relaxed);
    stats.selected_hits.store(0, std::memory_order_relaxed);
    stats.launch_completed_hits.store(0, std::memory_order_relaxed);
    stats.fallbacks.store(0, std::memory_order_relaxed);
    stats.circuit_breaker_activations.store(0, std::memory_order_relaxed);
    for (auto& reason : stats.reasons) {
        reason.store(0, std::memory_order_relaxed);
    }
}

bool fail_closed_trace_capture_active(const Mode mode, const std::optional<bool> observed_active) noexcept {
    return mode != Mode::Off && observed_active.value_or(true);
}

SelectedExecutionGuard::SelectedExecutionGuard(const bool* selected) noexcept :
    selected_(selected), uncaught_exceptions_(std::uncaught_exceptions()) {}

SelectedExecutionGuard::~SelectedExecutionGuard() noexcept {
    if (selected_ == nullptr || !*selected_) {
        return;
    }
    if (std::uncaught_exceptions() > uncaught_exceptions_) {
        circuit_break();
    } else {
        record_launch_completed_hit();
    }
}

}  // namespace ttnn::experimental::all_gather_minimal_matmul_registry
