// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/matmul_config_registry.hpp"

#include "ttnn/operation.hpp"

namespace ttnn::operations::matmul::registry {
namespace {

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
    if (resolution.reason != ResolutionReason::CertifiedMatch || !resolution.recipe.has_value()) {
        return ExecutionAction::Fallback;
    }
    if (mode == Mode::On) {
        return ExecutionAction::ApplyRecipe;
    }
    return mode == Mode::Shadow ? ExecutionAction::ObserveOnly : ExecutionAction::Fallback;
}

bool has_consistent_untilize_out(const Recipe& recipe) noexcept {
    if (const auto* config = std::get_if<MatmulMultiCoreReuseMultiCast1DProgramConfig>(&recipe.program_config)) {
        return config->untilize_out == recipe.untilize_out;
    }
    return !recipe.untilize_out;
}

Resolution resolve(const Mode mode, const Eligibility& eligibility) noexcept {
    if (mode == Mode::Off) {
        return {.reason = ResolutionReason::Disabled};
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
        eligibility.has_global_cb || eligibility.has_sub_device || eligibility.input_a_sharded ||
        eligibility.input_b_sharded || eligibility.input_b_batched || unsupported_transpose) {
        return {.reason = ResolutionReason::UnsupportedSemantics};
    }

    // The generated certified table intentionally starts empty. Shadow and On
    // therefore both preserve the existing selector and cache identity.
    return {.reason = ResolutionReason::EmptyRegistry};
}

}  // namespace ttnn::operations::matmul::registry
