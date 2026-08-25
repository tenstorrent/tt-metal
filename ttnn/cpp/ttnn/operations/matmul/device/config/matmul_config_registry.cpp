// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/matmul_config_registry.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
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
    std::atomic<std::uint64_t> applied_hits{0};
    std::atomic<std::uint64_t> fallbacks{0};
    std::atomic<std::uint64_t> circuit_breaker_activations{0};
    std::array<std::atomic<std::uint64_t>, kResolutionReasonCount> reasons{};
};

std::array<AtomicDomainStats, kOperationDomainCount> stats;
std::array<std::atomic<bool>, kOperationDomainCount> circuit_breakers{};

constexpr std::size_t index(const OperationDomain domain) { return static_cast<std::size_t>(domain); }
constexpr std::size_t index(const ResolutionReason reason) { return static_cast<std::size_t>(reason); }

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

std::optional<compact::KeyDescriptor> compact_key(const MatmulRegistryRequest& request) noexcept {
    const auto input_a = compact_tensor(request.input_a);
    const auto input_b = compact_tensor(request.input_b);
    const auto output = compact_tensor(request.output);
    const auto& device = request.device;
    if (!input_a.has_value() || !input_b.has_value() || !output.has_value() ||
        request.call.domain != OperationDomain::DenseMatmul ||
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
        .codegen_recipe_abi = 1,
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
        .domain = compact::Domain::DenseMatmul,
        .alpha_f32_bits = request.call.alpha_f32_bits.value_or(0),
        .beta_f32_bits = request.call.beta_f32_bits.value_or(0)};
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
    if (expected.build_identity_sha256 != actual.build_identity_sha256) {
        return CompatibilityStatus::BuildIdentityMismatch;
    }
    if (expected.runtime_capability_sha256 != actual.runtime_capability_sha256) {
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
    return initialize_registry_compatibility_for_testing(generated::metadata(), generated::entries().size(), actual);
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
    if (resolution.reason != ResolutionReason::CertifiedMatch || (!has_native_recipe && !has_compact_recipe)) {
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
    materialized.compute_kernel_config = selected_recipe->compute_kernel_config;
    materialized.untilize_out = selected_recipe->untilize_out;
    return materialized;
}

bool has_consistent_untilize_out(const Recipe& recipe) noexcept {
    if (const auto* config = std::get_if<MatmulMultiCoreReuseMultiCast1DProgramConfig>(&recipe.program_config)) {
        return config->untilize_out == recipe.untilize_out;
    }
    return !recipe.untilize_out;
}

MaterializationResult materialize_matmul_registry_recipe(const compact::EntryDescriptor& descriptor) {
    if (descriptor.key.schema_version != 1 || descriptor.key.codegen_recipe_abi != 1 ||
        descriptor.replay.schema_version != 2 || descriptor.key.domain != compact::Domain::DenseMatmul) {
        return {.status = MaterializationStatus::UnsupportedSchema};
    }
    if (descriptor.replay.family != compact::ProgramFamily::MultiCoreReuse) {
        return {.status = MaterializationStatus::UnsupportedReplay};
    }

    const auto& program = descriptor.replay.program_config;
    if (program.compute_grid_x == 0 || program.compute_grid_y == 0 || program.in0_block_w == 0 ||
        program.out_subblock_h == 0 || program.out_subblock_w == 0 || program.per_core_m == 0 ||
        program.per_core_n == 0 || program.allowed_worker_cores_present ||
        program.compute_grid_x > descriptor.key.compute_grid_x ||
        program.compute_grid_y > descriptor.key.compute_grid_y || program.out_subblock_h > 8 / program.out_subblock_w) {
        return {.status = MaterializationStatus::InvalidProgramConfig};
    }

    const auto& state = descriptor.replay.call_state;
    const auto& key = descriptor.key;
    const bool invalid_key_envelope =
        key.bcast_batch_present || key.bcast_batch || key.has_activation || key.has_bias || key.run_batched ||
        key.transpose_a || key.transpose_b || key.untilize_out || key.alpha_f32_bits != 0 || key.beta_f32_bits != 0 ||
        key.output.buffer_type != compact::BufferType::Dram || key.output.layout != compact::Layout::Tile ||
        key.output.memory_layout != compact::MemoryLayout::Interleaved || key.output.tile_height != 32 ||
        key.output.tile_width != 32;
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

static Resolution resolve_impl(
    const Mode mode,
    const MatmulRegistryRequest& request,
    const Eligibility& eligibility,
    const MatmulRegistryRequest* synthetic_request,
    const Recipe* synthetic_recipe,
    const compact::TableMetadata& metadata,
    const std::span<const compact::EntryDescriptor> entries,
    const CompatibilityDigests* actual_compatibility) noexcept {
    if (mode == Mode::Off) {
        return {.reason = ResolutionReason::Disabled};
    }
    if (mode == Mode::On && eligibility.trace_capture_active) {
        return {.reason = ResolutionReason::TraceCaptureUnsupported};
    }
    if (request.schema_version != 1) {
        return {.reason = ResolutionReason::IncompleteRequest};
    }
    const auto activation_end =
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
    if (is_domain_circuit_broken(request.call.domain)) {
        return {.reason = ResolutionReason::CircuitBroken};
    }
    if (request.call.domain != OperationDomain::DenseMatmul) {
        return {.reason = ResolutionReason::EmptyRegistry};
    }

    if (synthetic_request != nullptr && synthetic_recipe != nullptr && *synthetic_request == request) {
        return {.reason = ResolutionReason::CertifiedMatch, .recipe = synthetic_recipe};
    }

    if (entries.empty()) {
        return {.reason = ResolutionReason::EmptyRegistry};
    }
    if (request.device.attestation_status != DeviceAttestationStatus::Success) {
        return {.reason = ResolutionReason::DeviceAttestationUnavailable};
    }
    const auto compatibility = actual_compatibility != nullptr
                                   ? validate_registry_compatibility(metadata, entries.size(), *actual_compatibility)
                                   : startup_compatibility_status();
    if (compatibility != CompatibilityStatus::Compatible) {
        return {.reason = compatibility_reason(compatibility)};
    }
    if (metadata.runtime_capability_sha256 != request.device.runtime_capability_sha256 ||
        (actual_compatibility != nullptr &&
         actual_compatibility->runtime_capability_sha256 != request.device.runtime_capability_sha256)) {
        return {.reason = ResolutionReason::RuntimeCapabilityMismatch};
    }

    const auto key = compact_key(request);
    if (!key.has_value()) {
        return {.reason = ResolutionReason::IncompleteRequest};
    }
    if (const auto* descriptor = compact::ExactIndex{entries}.lookup(*key); descriptor != nullptr) {
        return {.reason = ResolutionReason::CertifiedMatch, .descriptor = descriptor};
    }

    return {.reason = ResolutionReason::EmptyRegistry};
}

Resolution resolve(const Mode mode, const MatmulRegistryRequest& request, const Eligibility& eligibility) noexcept {
    return resolve_impl(
        mode, request, eligibility, nullptr, nullptr, generated::metadata(), generated::entries(), nullptr);
}

DispatchResult resolve_for_dispatch(
    const Mode mode,
    const std::optional<MatmulRegistryRequest>& request,
    const Eligibility& eligibility,
    const ttnn::prim::MatmulParams& legacy_parameters,
    const ResolverFunction resolver) {
    auto resolution = Resolution{.reason = ResolutionReason::Disabled};
    if (mode == Mode::On && eligibility.trace_capture_active) {
        resolution = {.reason = ResolutionReason::TraceCaptureUnsupported};
    } else if (mode != Mode::Off) {
        resolution = request.has_value() && resolver != nullptr
                         ? resolver(mode, request.value(), eligibility)
                         : Resolution{.reason = ResolutionReason::IncompleteRequest};
    }

    auto action = execution_action(mode, resolution);
    std::optional<ttnn::prim::MatmulParams> materialized_parameters;
    if (action == ExecutionAction::ApplyRecipe && resolution.descriptor != nullptr) {
        auto materialized_recipe = materialize_matmul_registry_recipe(*resolution.descriptor);
        if (materialized_recipe.status != MaterializationStatus::Success || !materialized_recipe.recipe.has_value()) {
            const bool unsupported = materialized_recipe.status == MaterializationStatus::UnsupportedSchema ||
                                     materialized_recipe.status == MaterializationStatus::UnsupportedReplay;
            resolution.reason =
                unsupported ? ResolutionReason::UnsupportedReplay : ResolutionReason::MaterializationRejected;
            action = ExecutionAction::Fallback;
            circuit_break_domain(eligibility.call.domain);
        } else {
            const auto native_resolution =
                Resolution{.reason = ResolutionReason::CertifiedMatch, .recipe = &materialized_recipe.recipe.value()};
            materialized_parameters = materialize_parameters_for_execution(mode, native_resolution, legacy_parameters);
        }
    } else {
        materialized_parameters = materialize_parameters_for_execution(mode, resolution, legacy_parameters);
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
        nullptr);
}

Resolution resolve_with_compact_table_for_testing(
    const Mode mode,
    const MatmulRegistryRequest& request,
    const Eligibility& eligibility,
    const compact::TableMetadata& metadata,
    const std::span<const compact::EntryDescriptor> entries,
    const CompatibilityDigests& actual) noexcept {
    return resolve_impl(mode, request, eligibility, nullptr, nullptr, metadata, entries, &actual);
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
        (resolution.recipe != nullptr || resolution.descriptor != nullptr)) {
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
    snapshot.compatibility_status = startup_compatibility_status();
    snapshot.table_metadata = generated::metadata();
    snapshot.entry_count = generated::entries().size();

    for (std::size_t domain = 0; domain < stats.size(); ++domain) {
        const auto& source = stats[domain];
        auto& destination = snapshot.domains[domain];
        destination.resolution_attempts = source.resolution_attempts.load(std::memory_order_relaxed);
        destination.certified_hits = source.certified_hits.load(std::memory_order_relaxed);
        destination.shadow_would_hits = source.shadow_would_hits.load(std::memory_order_relaxed);
        destination.applied_hits = source.applied_hits.load(std::memory_order_relaxed);
        destination.fallbacks = source.fallbacks.load(std::memory_order_relaxed);
        destination.circuit_breaker_activations = source.circuit_breaker_activations.load(std::memory_order_relaxed);
        destination.circuit_broken = circuit_breakers[domain].load(std::memory_order_acquire);
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
        domain.circuit_breaker_activations.store(0, std::memory_order_relaxed);
        for (auto& reason : domain.reasons) {
            reason.store(0, std::memory_order_relaxed);
        }
    }
}

}  // namespace ttnn::operations::matmul::registry
