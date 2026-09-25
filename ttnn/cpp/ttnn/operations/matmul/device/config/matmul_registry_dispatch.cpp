// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_registry_dispatch.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <utility>

#include <tt-metalium/experimental/inspector.hpp>

#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

namespace ttnn::operations::matmul::registry {

bool has_only_unit_front_dimensions(const ttnn::Shape& shape) noexcept {
    if (shape.rank() < 2) {
        return false;
    }
    for (std::size_t index = 0; index + 2 < shape.rank(); ++index) {
        if (shape[index] != 1) {
            return false;
        }
    }
    return true;
}

RegistryRequestInspection inspect_registry_request(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const bool has_bias,
    const CallSemantics call_semantics,
    const ttnn::prim::MatmulParams& parameters,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    const bool trace_capture_active) {
    const auto io_contract = resolve_matmul_io_contract(IoContractRequest{
        .input_a_dtype = input_tensor_a.dtype(),
        .input_a_tile = input_tensor_a.tensor_spec().tile(),
        .input_b_tile = input_tensor_b.tensor_spec().tile(),
        .requested_output_memory_config = parameters.output_mem_config,
        .requested_output_dtype = parameters.output_dtype,
        .requested_output_tile = parameters.output_tile,
        .optional_output = optional_output_tensor.has_value()
                               ? std::make_optional(OptionalOutputContract{
                                     .memory_config = optional_output_tensor->memory_config(),
                                     .dtype = optional_output_tensor->dtype(),
                                     .tile = optional_output_tensor->tensor_spec().tile()})
                               : std::nullopt,
        .transpose_a = parameters.transpose_a,
        .transpose_b = parameters.transpose_b,
    });
    RegistryRequestInspection inspection{
        .eligibility = v1_eligibility_from_call_state(
            call_semantics,
            io_contract.status,
            trace_capture_active,
            has_bias,
            parameters,
            optional_output_tensor.has_value(),
            input_tensor_a.is_sharded(),
            input_tensor_b.is_sharded(),
            io_contract.output_memory_config.is_sharded(),
            has_nondefault_v1_tile_transpose(input_tensor_a.tensor_spec().tile()) ||
                has_nondefault_v1_tile_transpose(input_tensor_b.tensor_spec().tile()) ||
                has_nondefault_v1_tile_transpose(io_contract.output_tile) ||
                input_tensor_a.tensor_spec().tile().get_height() != 32 ||
                input_tensor_a.tensor_spec().tile().get_width() != 32 ||
                input_tensor_b.tensor_spec().tile().get_height() != 32 ||
                input_tensor_b.tensor_spec().tile().get_width() != 32 || io_contract.output_tile.get_height() != 32 ||
                io_contract.output_tile.get_width() != 32)};
    // Reject caller-known exclusions before shape and device inspection.
    if (preflight_v1_eligibility(inspection.eligibility) != ResolutionReason::CertifiedMatch) {
        return inspection;
    }

    const auto* device_a = input_tensor_a.device();
    const auto* device_b = input_tensor_b.device();
    // This registry contains local dense-matmul recipes only. Distributed
    // matmul families (for example AGMM) own separate exact tables keyed by
    // their collective topology; a mesh-wide tensor must never consume this
    // single-device evidence merely because each local launch is legal.
    if (!has_only_unit_front_dimensions(input_tensor_a.logical_shape()) ||
        !has_only_unit_front_dimensions(input_tensor_b.logical_shape()) ||
        device_a == nullptr || device_a != device_b || device_a->num_devices() != 1) {
        return inspection;
    }
    // MeshDevice::arch() currently consults the default MetalContext. Use the
    // mesh's physical device so non-default MetalEnv instances cannot be
    // mistaken for the architecture that owns the checked-in evidence.
    const auto devices = device_a->get_devices();
    if (devices.size() != 1 || devices.front() == nullptr || devices.front()->arch() != tt::ARCH::BLACKHOLE) {
        return inspection;
    }
    const auto device_arch = devices.front()->arch();

    const auto a_logical = utilities::get_matmul_tensor_logical_shape(input_tensor_a, parameters.transpose_a);
    const auto b_logical = utilities::get_matmul_tensor_logical_shape(input_tensor_b, parameters.transpose_b);
    const auto a_padded = utilities::get_matmul_tensor_padded_shape(input_tensor_a, parameters.transpose_a);
    const auto b_padded = utilities::get_matmul_tensor_padded_shape(input_tensor_b, parameters.transpose_b);
    if (a_logical[-1] != b_logical[-2] || a_padded[-1] != b_padded[-2]) {
        return inspection;
    }

    const auto tensor_request = [](const ttnn::Tensor& tensor) {
        const auto& tile = tensor.tensor_spec().tile();
        const auto& memory_config = tensor.memory_config();
        return TensorRequest{
            .dtype = tensor.dtype(),
            .layout = tensor.layout(),
            .memory_layout = memory_config.memory_layout(),
            .buffer_type = memory_config.buffer_type(),
            .tile_height = tile.get_height(),
            .tile_width = tile.get_width(),
        };
    };
    const auto grid = device_a->compute_with_storage_grid_size();
    std::optional<std::uint32_t> activation_op;
    std::array<std::uint32_t, MatmulRegistryRequest::kMaxActivationParameters> activation_params{};
    std::uint8_t activation_param_count = 0;
    if (parameters.user_fused_activation.has_value()) {
        activation_op = static_cast<std::uint32_t>(parameters.user_fused_activation->op_type);
        if (parameters.user_fused_activation->params.size() > activation_params.size()) {
            return inspection;
        }
        for (const auto parameter : parameters.user_fused_activation->params) {
            activation_params[activation_param_count++] = std::bit_cast<std::uint32_t>(parameter);
        }
    }

    std::optional<compact::ComputeKernelDescriptor> compute_kernel_config;
    if (parameters.compute_kernel_config.has_value()) {
        compute_kernel_config = compact_compute_kernel_config(*parameters.compute_kernel_config);
        if (!compute_kernel_config.has_value()) {
            // Unspellable in the compact key: stay on the legacy path.
            return inspection;
        }
    }

    inspection.request = MatmulRegistryRequest{
        .schema_version = compact::kKeySchemaVersion,
        .call = call_semantics,
        .workload =
            WorkloadRequest{
                .logical_m = a_logical[-2],
                .logical_k = a_logical[-1],
                .logical_n = b_logical[-1],
                .padded_m = a_padded[-2],
                .padded_k = a_padded[-1],
                .padded_n = b_padded[-1],
            },
        .input_a = tensor_request(input_tensor_a),
        .input_b = tensor_request(input_tensor_b),
        .output =
            TensorRequest{
                .dtype = io_contract.output_dtype,
                .layout = tt::tt_metal::Layout::TILE,
                .memory_layout = io_contract.output_memory_config.memory_layout(),
                .buffer_type = io_contract.output_memory_config.buffer_type(),
                .tile_height = io_contract.output_tile.get_height(),
                .tile_width = io_contract.output_tile.get_width(),
            },
        .device =
            DeviceRequest{
                .architecture = static_cast<std::uint32_t>(device_arch),
                .device_count = static_cast<std::uint32_t>(device_a->num_devices()),
                .mesh_rows = static_cast<std::uint32_t>(device_a->num_rows()),
                .mesh_cols = static_cast<std::uint32_t>(device_a->num_cols()),
                .compute_grid_x = grid.x,
                .compute_grid_y = grid.y,
            },
        .transpose_a = parameters.transpose_a,
        .transpose_b = parameters.transpose_b,
        .has_bias = has_bias,
        .has_activation = parameters.user_fused_activation.has_value(),
        .untilize_out = parameters.untilize_out,
        .bcast_batch = parameters.bcast_batch,
        .run_batched = parameters.user_run_batched,
        .activation_op = activation_op,
        .activation_param_f32_bits = activation_params,
        .activation_param_count = activation_param_count,
        .compute_kernel_config = compute_kernel_config,
    };
    return inspection;
}

void try_apply_registry_parameters(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const bool has_bias,
    const CallSemantics call_semantics,
    ttnn::prim::MatmulParams& parameters,
    const std::optional<ttnn::Tensor>& optional_output_tensor) {
    // This function runs entirely in host-side dispatch before a device
    // operation or kernel is launched. Strict-mode exceptions are therefore
    // ordinary host validation failures, never device-code exceptions.
    const auto mode = current_mode();
    if (mode == Mode::Off) {
        return;
    }

    RegistryRequestInspection inspection;
    try {
        // Caller-known exclusions are deliberately inspected before the trace
        // API so unsupported calls stay on the cheapest legacy path.
        inspection = inspect_registry_request(
            input_tensor_a,
            input_tensor_b,
            has_bias,
            call_semantics,
            parameters,
            optional_output_tensor,
            /*trace_capture_active=*/false);
    } catch (...) {
        // Request construction may inspect tensors before legacy validators
        // do. Preserve their exception timing by falling back untouched.
        if (fallback_is_error(mode)) {
            throw;
        }
        const Eligibility eligibility{.call = call_semantics};
        static_cast<void>(resolve_for_dispatch(mode, std::nullopt, eligibility, parameters));
        return;
    }
    if (!inspection.request.has_value()) {
        static_cast<void>(resolve_for_dispatch(mode, std::nullopt, inspection.eligibility, parameters));
        return;
    }

    try {
        auto* device = input_tensor_a.device();
        if (device == nullptr || tt::tt_metal::experimental::inspector::GetCurrentMeshTraceId(device).has_value()) {
            inspection.eligibility.trace_capture_active = true;
            static_cast<void>(resolve_for_dispatch(mode, std::nullopt, inspection.eligibility, parameters));
            return;
        }
    } catch (...) {
        if (fallback_is_error(mode)) {
            throw;
        }
        // An unavailable trace state is indistinguishable from active capture.
        inspection.eligibility.trace_capture_active = true;
        static_cast<void>(resolve_for_dispatch(mode, std::nullopt, inspection.eligibility, parameters));
        return;
    }

    auto dispatch = resolve_for_dispatch(mode, inspection.request, inspection.eligibility, parameters);
    if (dispatch.action != ExecutionAction::ApplyRecipe || !dispatch.materialized_parameters.has_value()) {
        return;
    }

    auto caller_compute_kernel_config = parameters.compute_kernel_config;
    try {
        // Preflight proved program_config absent, and lookup proved any
        // caller CKC equals the entry's, so this assignment is either a fill
        // or a no-op. A partial failure restores the caller's original state.
        parameters.program_config = std::move(dispatch.materialized_parameters->program_config);
        parameters.compute_kernel_config = dispatch.materialized_parameters->compute_kernel_config;
        return;
    } catch (...) {
        parameters.program_config.reset();
        // ComputeKernelConfig is six scalars: trivially copyable, so a move here
        // is a copy wearing a costume. clang-tidy performance-move-const-arg.
        parameters.compute_kernel_config = caller_compute_kernel_config;
        if (fallback_is_error(mode)) {
            throw;
        }
        return;
    }
}

}  // namespace ttnn::operations::matmul::registry
