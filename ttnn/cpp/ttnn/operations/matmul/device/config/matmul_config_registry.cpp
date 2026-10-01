// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_config_registry.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <string_view>
#include <utility>

#include <tt_stl/assert.hpp>

#include "matmul_registry_data.hpp"

namespace ttnn::operations::matmul::registry {
namespace {

ExecutionAction execution_action(const Mode mode, const Resolution& resolution) noexcept {
    if (resolution.reason != ResolutionReason::CertifiedMatch) {
        return ExecutionAction::Fallback;
    }
    if (mode == Mode::On) {
        return ExecutionAction::ApplyRecipe;
    }
    return mode == Mode::Shadow ? ExecutionAction::ObserveOnly : ExecutionAction::Fallback;
}

constexpr std::string_view reason_name(const ResolutionReason reason) noexcept {
    switch (reason) {
        case ResolutionReason::Disabled: return "disabled";
        case ResolutionReason::IneligibleOperationDomain: return "ineligible operation domain";
        case ResolutionReason::MalformedOperationSemantics: return "malformed operation semantics";
        case ResolutionReason::InconsistentIoContract: return "inconsistent I/O contract";
        case ResolutionReason::TraceCaptureUnsupported: return "trace capture unsupported";
        case ResolutionReason::ExplicitOverride: return "explicit override";
        case ResolutionReason::UnsupportedSemantics: return "unsupported semantics";
        case ResolutionReason::IncompleteRequest: return "incomplete request";
        case ResolutionReason::InconsistentRequest: return "inconsistent request";
        case ResolutionReason::MaterializationRejected: return "materialization rejected";
        case ResolutionReason::EmptyRegistry: return "no exact match";
        case ResolutionReason::CertifiedMatch: return "certified match";
    }
    return "unknown reason";
}

std::optional<compact::DataType> compact_dtype(const tt::tt_metal::DataType dtype) noexcept {
    switch (dtype) {
        case tt::tt_metal::DataType::BFLOAT16: return compact::DataType::BFloat16;
        case tt::tt_metal::DataType::BFLOAT8_B: return compact::DataType::BFloat8B;
        case tt::tt_metal::DataType::FLOAT32: return compact::DataType::Float32;
        case tt::tt_metal::DataType::BFLOAT4_B: return compact::DataType::BFloat4B;
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
    if (!dtype || !layout || !buffer_type || tensor.memory_layout != tt::tt_metal::TensorMemoryLayout::INTERLEAVED ||
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

// The compute-kernel knobs a matmul call that supplies none actually runs at,
// mirrored from create_matmul_attributes() and init_device_compute_kernel_config()
// in matmul_device_operation.cpp. The key has to name a knob vector, and for a
// default call that vector is not unknown -- it is this one, so this is what it
// is keyed at. Preflight has already rejected a caller program config and a
// caller core grid, which is exactly the state in which create_matmul_attributes
// raises fidelity, so the has_program_config/has_user_grid terms of its rule are
// constant here and are not repeated.
compact::ComputeKernelDescriptor default_compute_kernel(const MatmulRegistryRequest& request) noexcept {
    const auto is_low_precision = [](const tt::tt_metal::DataType dtype) {
        return dtype == tt::tt_metal::DataType::BFLOAT8_B || dtype == tt::tt_metal::DataType::BFLOAT4_B;
    };
    const bool inputs_low_precision =
        is_low_precision(request.input_a.dtype) && is_low_precision(request.input_b.dtype);
    const bool inputs_float32 = request.input_a.dtype == tt::tt_metal::DataType::FLOAT32 &&
                                request.input_b.dtype == tt::tt_metal::DataType::FLOAT32;
    auto fidelity = inputs_low_precision ? compact::MathFidelity::LoFi : compact::MathFidelity::HiFi2;
    if (inputs_float32) {
        // Hardware bug #38306: HiFi4 with fp32 accumulation can be wrong on
        // Wormhole, so TTNN drops to HiFi3 there and only there.
        fidelity = request.device.architecture == static_cast<std::uint32_t>(tt::ARCH::WORMHOLE_B0)
                       ? compact::MathFidelity::HiFi3
                       : compact::MathFidelity::HiFi4;
    }
    const bool output_float32 = request.output.dtype == tt::tt_metal::DataType::FLOAT32;
    return compact::ComputeKernelDescriptor{
        .math_fidelity = fidelity,
        .throttle_level = compact::ThrottleLevel::NoThrottle,
        .math_approx_mode = compact::kMathApproxModeIsInertAt,
        .fp32_dest_acc_en = output_float32,
        .packer_l1_acc = !output_float32,
        .dst_full_sync_en = false};
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

Resolution resolve_from_tables(
    const MatmulRegistryRequest& request,
    const Eligibility& eligibility,
    const std::span<const compact::ProgramConfigExactEntry> exact_entries) noexcept {
    const auto envelope_reason = validate_v1_request_envelope(request, eligibility);
    if (envelope_reason != ResolutionReason::CertifiedMatch) {
        return {.reason = envelope_reason};
    }
    if (exact_entries.empty()) {
        return {.reason = ResolutionReason::EmptyRegistry};
    }
    const auto key = compact_registry_key(request);
    if (!key) {
        return {.reason = ResolutionReason::IncompleteRequest};
    }

    // Exact entries are harvested-grid cohorts. Never erase the live grid:
    // distinct 11/12/13-column measurements may coexist and must not shadow
    // one another.
    const auto* exact = compact::lookup_program_config_exact(*key, exact_entries);
    if (exact == nullptr &&
        (request.call.domain == OperationDomain::Linear || request.call.domain == OperationDomain::Addmm)) {
        // The admitted linear envelope has no bias or activation, and the
        // admitted addmm envelope has alpha=1 and beta=+/-0. Both therefore
        // execute the same inner matmul as dense.matmul. Prefer a future
        // operation-specific recipe above, but let today's dense measurements
        // serve these provably kernel-equivalent public wrappers.
        auto dense_key = *key;
        dense_key.domain = compact::Domain::DenseMatmul;
        dense_key.alpha_f32_bits = 0;
        dense_key.beta_f32_bits = 0;
        // key.compute_kernel is deliberately carried across unchanged: only the
        // operation domain is being widened here, never the numerics.
        exact = compact::lookup_program_config_exact(dense_key, exact_entries);
    }
    if (exact != nullptr) {
        const compact::ProgramConfigCandidate candidate{
            .program_config = exact->program_config, .compute_kernel_config = exact->compute_kernel_config};
        if (!compact::legal_program_config_candidate(*key, candidate)) {
            return {.reason = ResolutionReason::MaterializationRejected};
        }
        return {
            .reason = ResolutionReason::CertifiedMatch,
            .program_config = exact->program_config,
            .compute_kernel_config = exact->compute_kernel_config,
            .key = key};
    }

    return {.reason = ResolutionReason::EmptyRegistry};
}

}  // namespace

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
    if (request.optional_output) {
        const auto& output = *request.optional_output;
        if (output_memory_config == tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
            output_memory_config = output.memory_config;
        } else if (output_memory_config != output.memory_config) {
            return {
                IoContractStatus::OptionalOutputMemoryMismatch,
                output_memory_config,
                output_dtype,
                request.input_a_tile,
                true};
        }
        if (request.requested_output_dtype && *request.requested_output_dtype != output.dtype) {
            return {
                IoContractStatus::OptionalOutputDtypeMismatch,
                output_memory_config,
                output_dtype,
                request.input_a_tile,
                true};
        }
        output_dtype = output.dtype;
    }

    const auto input_a_tile = transpose_matmul_tile(request.input_a_tile, request.transpose_a);
    const auto input_b_tile = transpose_matmul_tile(request.input_b_tile, request.transpose_b);
    if (!input_a_tile || !input_b_tile) {
        return {
            IoContractStatus::InvalidTransposeTile,
            output_memory_config,
            output_dtype,
            request.input_a_tile,
            request.optional_output.has_value()};
    }
    if (request.requested_output_tile && request.optional_output) {
        return {
            IoContractStatus::OutputTileConflict,
            output_memory_config,
            output_dtype,
            *request.requested_output_tile,
            true};
    }
    const auto output_tile = request.requested_output_tile ? *request.requested_output_tile
                             : request.optional_output
                                 ? request.optional_output->tile
                                 : tt::tt_metal::Tile({input_a_tile->get_height(), input_b_tile->get_width()});
    return {
        IoContractStatus::Resolved,
        output_memory_config,
        output_dtype,
        output_tile,
        request.optional_output.has_value()};
}

std::optional<compact::KeyDescriptor> compact_registry_key(const MatmulRegistryRequest& request) noexcept {
    const auto input_a = compact_tensor(request.input_a);
    const auto input_b = compact_tensor(request.input_b);
    const auto output = compact_tensor(request.output);
    const auto domain = compact_domain(request.call.domain);
    const auto& device = request.device;
    if (!input_a || !input_b || !output || !domain || device.device_count > std::numeric_limits<std::uint16_t>::max() ||
        device.mesh_rows > std::numeric_limits<std::uint16_t>::max() ||
        device.mesh_cols > std::numeric_limits<std::uint16_t>::max() ||
        device.compute_grid_x > std::numeric_limits<std::uint16_t>::max() ||
        device.compute_grid_y > std::numeric_limits<std::uint16_t>::max()) {
        return std::nullopt;
    }
    return compact::normalize_key_compute_kernel(compact::KeyDescriptor{
        .architecture = device.architecture,
        .bcast_batch_present = request.bcast_batch.has_value(),
        .bcast_batch = request.bcast_batch.value_or(false),
        .board_capability_class = 0,
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
        .transpose_a = request.transpose_a,
        .transpose_b = request.transpose_b,
        .untilize_out = request.untilize_out,
        .domain = *domain,
        .alpha_f32_bits = request.call.alpha_f32_bits.value_or(0),
        .beta_f32_bits = request.call.beta_f32_bits.value_or(0),
        // A caller-supplied config is what the kernel will run at, so it is what
        // the table is asked about; absent one, the call runs at TTNN's own
        // defaults, so those are.
        .compute_kernel = request.compute_kernel_config.value_or(default_compute_kernel(request))});
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
    if ((is_addmm && (!has_alpha || !has_beta)) || (!is_addmm && (has_alpha || has_beta)) ||
        (is_addmm && *eligibility.call.alpha_f32_bits != 0x3F800000U)) {
        return ResolutionReason::MalformedOperationSemantics;
    }
    if (eligibility.io_contract_status != IoContractStatus::Resolved) {
        return ResolutionReason::InconsistentIoContract;
    }
    // A caller-supplied compute kernel config is keyed, not declined: it is
    // matched against the entry's measured CKC during lookup. Program config
    // and core grid remain registry-owned axes we must not override.
    if (eligibility.has_program_config || eligibility.has_user_core_grid) {
        return ResolutionReason::ExplicitOverride;
    }
    // Decline only what the key cannot tell apart. A returned entry carries a
    // program config, so serving a call is safe exactly when every call that maps
    // to one key needs one config. Two distinct reasons were previously merged
    // into a single condition, which cost coverage for no safety:
    //
    //   * these are UNDER-KEYED, so distinct calls would collide on one entry and
    //     silently receive a recipe measured for the other. They must be declined.
    //       has_activation  -- the key stores only a bool, never which activation,
    //                          so relu and gelu produce an identical key.
    //       *_sharded       -- MemoryLayout has a single enumerator, Interleaved,
    //                          so a shard spec is not representable at all.
    //       global_cb, sub_device, optional_output, output_tile,
    //       unsupported_tile_metadata -- not represented in the key in any form.
    //       has_bias        -- keyed as a bool only; bias length follows N, so this
    //                          is probably safe to admit, but "probably" is not a
    //                          basis for handing out a config. Left declined until
    //                          measured.
    if (eligibility.has_bias || eligibility.has_activation || eligibility.has_unsupported_tile_metadata ||
        eligibility.has_optional_output || eligibility.has_output_tile || eligibility.has_global_cb ||
        eligibility.has_sub_device || eligibility.input_a_sharded || eligibility.input_b_sharded ||
        eligibility.output_sharded ||
        (is_addmm && *eligibility.call.beta_f32_bits != 0 && *eligibility.call.beta_f32_bits != 0x80000000U)) {
        return ResolutionReason::UnsupportedSemantics;
    }
    //   * transpose_a, transpose_b, untilize_out, bcast_batch and run_batched are
    //     FULLY KEYED (KeyDescriptor fields of the same name; run_batched carries
    //     input_b_batched and is re-checked in validate_v1_request_envelope). A
    //     call that sets one keys to its own entry, so a collision is impossible
    //     and declining it bought nothing -- it only guaranteed we could never
    //     serve those shapes even once measured. They now fall through to lookup:
    //     absent measurements they simply miss and fall back, which is the same
    //     outcome the fence produced, and once swept they are served correctly.
    return ResolutionReason::CertifiedMatch;
}

ResolutionReason validate_v1_request_envelope(
    const MatmulRegistryRequest& request, const Eligibility& eligibility) noexcept {
    const auto preflight = preflight_v1_eligibility(eligibility);
    if (preflight != ResolutionReason::CertifiedMatch) {
        return preflight;
    }
    if (request.schema_version != compact::kKeySchemaVersion) {
        return ResolutionReason::IncompleteRequest;
    }
    const auto parameter_count =
        std::min<std::size_t>(request.activation_param_count, request.activation_param_f32_bits.size());
    const bool nonzero_padding = std::any_of(
        request.activation_param_f32_bits.begin() + parameter_count,
        request.activation_param_f32_bits.end(),
        [](const auto value) { return value != 0; });
    if (request.call != eligibility.call || request.transpose_a != eligibility.transpose_a ||
        request.transpose_b != eligibility.transpose_b || request.has_bias != eligibility.has_bias ||
        request.has_activation != eligibility.has_activation || request.untilize_out != eligibility.untilize_out ||
        request.compute_kernel_config.has_value() != eligibility.has_compute_kernel_config ||
        request.bcast_batch.has_value() != eligibility.has_bcast_batch ||
        request.run_batched != eligibility.input_b_batched ||
        request.has_activation != request.activation_op.has_value() ||
        request.activation_param_count > request.activation_param_f32_bits.size() ||
        (!request.has_activation && request.activation_param_count != 0) || nonzero_padding) {
        return ResolutionReason::InconsistentRequest;
    }
    return ResolutionReason::CertifiedMatch;
}

Resolution resolve(const MatmulRegistryRequest& request, const Eligibility& eligibility) noexcept {
    return resolve_from_tables(request, eligibility, generated::program_config_exact_entries());
}

Resolution resolve_with_compact_table_for_testing(
    const MatmulRegistryRequest& request,
    const Eligibility& eligibility,
    const std::span<const compact::ProgramConfigExactEntry> exact_entries) noexcept {
    return resolve_from_tables(request, eligibility, exact_entries);
}

Mode current_mode() noexcept {
    const auto configured = ttnn::CONFIG.get<"matmul_registry_mode">();
    const auto configured_value = static_cast<std::uint8_t>(configured);
    return configured_value <= static_cast<std::uint8_t>(Mode::On) ? configured : Mode::Off;
}

bool fallback_is_error(const Mode mode) {
    return mode == Mode::On && ttnn::CONFIG.get<"throw_exception_on_fallback">();
}

std::optional<MatmulProgramConfig> materialize_registry_program_config(
    const compact::KeyDescriptor& key,
    const compact::ProgramConfigDescriptor& descriptor,
    const std::optional<compact::ComputeKernelDescriptor> compute_kernel_config) {
    if (!compute_kernel_config) {
        return std::nullopt;
    }
    const compact::ProgramConfigCandidate candidate{
        .program_config = descriptor, .compute_kernel_config = *compute_kernel_config};
    if (!compact::legal_program_config_candidate(key, candidate)) {
        return std::nullopt;
    }
    const auto grid = tt::tt_metal::CoreCoord{descriptor.compute_grid_x, descriptor.compute_grid_y};
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

std::optional<DeviceComputeKernelConfig> materialize_registry_compute_kernel_config(
    const compact::ComputeKernelDescriptor& descriptor) {
    tt::tt_metal::MathFidelity fidelity;
    switch (descriptor.math_fidelity) {
        case compact::MathFidelity::LoFi: fidelity = tt::tt_metal::MathFidelity::LoFi; break;
        case compact::MathFidelity::HiFi2: fidelity = tt::tt_metal::MathFidelity::HiFi2; break;
        case compact::MathFidelity::HiFi3: fidelity = tt::tt_metal::MathFidelity::HiFi3; break;
        case compact::MathFidelity::HiFi4: fidelity = tt::tt_metal::MathFidelity::HiFi4; break;
        default: return std::nullopt;
    }
    compute_throttle_utils::ThrottleLevel throttle;
    switch (descriptor.throttle_level) {
        case compact::ThrottleLevel::NoThrottle: throttle = compute_throttle_utils::ThrottleLevel::NO_THROTTLE; break;
        case compact::ThrottleLevel::Throttle1: throttle = compute_throttle_utils::ThrottleLevel::LEVEL_1; break;
        case compact::ThrottleLevel::Throttle2: throttle = compute_throttle_utils::ThrottleLevel::LEVEL_2; break;
        case compact::ThrottleLevel::Throttle3: throttle = compute_throttle_utils::ThrottleLevel::LEVEL_3; break;
        case compact::ThrottleLevel::Throttle4: throttle = compute_throttle_utils::ThrottleLevel::LEVEL_4; break;
        case compact::ThrottleLevel::Throttle5: throttle = compute_throttle_utils::ThrottleLevel::LEVEL_5; break;
        default: return std::nullopt;
    }
    return DeviceComputeKernelConfig{
        .math_fidelity = fidelity,
        .math_approx_mode = descriptor.math_approx_mode,
        .fp32_dest_acc_en = descriptor.fp32_dest_acc_en,
        .packer_l1_acc = descriptor.packer_l1_acc,
        .dst_full_sync_en = descriptor.dst_full_sync_en,
        .throttle_level = throttle};
}

std::optional<compact::ComputeKernelDescriptor> compact_compute_kernel_config(
    const DeviceComputeKernelConfig& config) noexcept {
    compact::MathFidelity fidelity;
    switch (config.math_fidelity) {
        case tt::tt_metal::MathFidelity::LoFi: fidelity = compact::MathFidelity::LoFi; break;
        case tt::tt_metal::MathFidelity::HiFi2: fidelity = compact::MathFidelity::HiFi2; break;
        case tt::tt_metal::MathFidelity::HiFi3: fidelity = compact::MathFidelity::HiFi3; break;
        case tt::tt_metal::MathFidelity::HiFi4: fidelity = compact::MathFidelity::HiFi4; break;
        default: return std::nullopt;
    }
    compact::ThrottleLevel throttle;
    switch (config.throttle_level) {
        case compute_throttle_utils::ThrottleLevel::NO_THROTTLE: throttle = compact::ThrottleLevel::NoThrottle; break;
        case compute_throttle_utils::ThrottleLevel::LEVEL_1: throttle = compact::ThrottleLevel::Throttle1; break;
        case compute_throttle_utils::ThrottleLevel::LEVEL_2: throttle = compact::ThrottleLevel::Throttle2; break;
        case compute_throttle_utils::ThrottleLevel::LEVEL_3: throttle = compact::ThrottleLevel::Throttle3; break;
        case compute_throttle_utils::ThrottleLevel::LEVEL_4: throttle = compact::ThrottleLevel::Throttle4; break;
        case compute_throttle_utils::ThrottleLevel::LEVEL_5: throttle = compact::ThrottleLevel::Throttle5; break;
        default: return std::nullopt;
    }
    return compact::ComputeKernelDescriptor{
        .math_fidelity = fidelity,
        .throttle_level = throttle,
        .math_approx_mode = config.math_approx_mode,
        .fp32_dest_acc_en = config.fp32_dest_acc_en,
        .packer_l1_acc = config.packer_l1_acc,
        .dst_full_sync_en = config.dst_full_sync_en};
}

std::optional<ttnn::prim::MatmulParams> materialize_parameters_for_execution(
    const Resolution& resolution, const ttnn::prim::MatmulParams& legacy_parameters) {
    if (resolution.reason != ResolutionReason::CertifiedMatch || !resolution.key || !resolution.program_config ||
        !resolution.compute_kernel_config) {
        return std::nullopt;
    }
    auto program_config = materialize_registry_program_config(
        *resolution.key, *resolution.program_config, resolution.compute_kernel_config);
    auto compute_kernel_config = materialize_registry_compute_kernel_config(*resolution.compute_kernel_config);
    if (!program_config || !compute_kernel_config) {
        return std::nullopt;
    }
    auto result = legacy_parameters;
    result.program_config = std::move(program_config);
    result.compute_kernel_config = compute_kernel_config;
    return result;
}

DispatchResult resolve_for_dispatch(
    const Mode mode,
    const std::optional<MatmulRegistryRequest>& request,
    const Eligibility& eligibility,
    const ttnn::prim::MatmulParams& legacy_parameters,
    const ResolverFunction resolver) {
    Resolution resolution{.reason = ResolutionReason::Disabled};
    if (mode != Mode::Off) {
        if (const auto preflight = preflight_v1_eligibility(eligibility);
                   preflight != ResolutionReason::CertifiedMatch) {
            resolution.reason = preflight;
        } else if (!request || resolver == nullptr) {
            resolution.reason = ResolutionReason::IncompleteRequest;
        } else {
            resolution = resolver(*request, eligibility);
        }
    }

    auto action = execution_action(mode, resolution);
    std::optional<ttnn::prim::MatmulParams> materialized;
    if (action == ExecutionAction::ApplyRecipe) {
        try {
            materialized = materialize_parameters_for_execution(resolution, legacy_parameters);
        } catch (...) {
            materialized.reset();
        }
        if (!materialized) {
            resolution.reason = ResolutionReason::MaterializationRejected;
            action = ExecutionAction::Fallback;
        }
    }
    if (fallback_is_error(mode) && action != ExecutionAction::ApplyRecipe) {
        TT_THROW(
            "Matmul registry required an exact recipe, but dispatch fell back: {}", reason_name(resolution.reason));
    }

    return {.resolution = resolution, .action = action, .materialized_parameters = std::move(materialized)};
}

std::optional<ttnn::prim::MatmulParams> select_registry_parameters(
    const Mode mode,
    const MatmulRegistryRequest& request,
    const Eligibility& eligibility,
    const ttnn::prim::MatmulParams& legacy_parameters) {
    return resolve_for_dispatch(mode, request, eligibility, legacy_parameters).materialized_parameters;
}

}  // namespace ttnn::operations::matmul::registry
