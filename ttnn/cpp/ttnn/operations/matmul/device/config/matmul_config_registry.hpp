// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include <tt-metalium/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tile.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/matmul/device/config/registry/matmul_registry_exact.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::operations::matmul::registry {

enum class OperationDomain : std::uint8_t { DenseMatmul, Linear, Addmm, IneligibleSharedCaller };

struct CallSemantics {
    OperationDomain domain = OperationDomain::IneligibleSharedCaller;
    std::optional<std::uint32_t> alpha_f32_bits = std::nullopt;
    std::optional<std::uint32_t> beta_f32_bits = std::nullopt;

    bool operator==(const CallSemantics&) const = default;
};

constexpr CallSemantics dense_matmul_call_semantics() noexcept {
    return CallSemantics{.domain = OperationDomain::DenseMatmul};
}
constexpr CallSemantics linear_call_semantics() noexcept { return CallSemantics{.domain = OperationDomain::Linear}; }
CallSemantics addmm_call_semantics(float alpha, float beta) noexcept;

enum class ResolutionReason : std::uint8_t {
    IneligibleOperationDomain,
    MalformedOperationSemantics,
    InconsistentIoContract,
    TraceCaptureUnsupported,
    ExplicitOverride,
    UnsupportedSemantics,
    IncompleteRequest,
    InconsistentRequest,
    UnsupportedArtifact,
    MaterializationRejected,
    EmptyRegistry,
    CertifiedMatch,
};

enum class IoContractStatus : std::uint8_t {
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
    IoContractStatus status = IoContractStatus::Resolved;
    tt::tt_metal::MemoryConfig output_memory_config;
    tt::tt_metal::DataType output_dtype;
    tt::tt_metal::Tile output_tile;
    bool uses_optional_output = false;
};

ResolvedMatmulIoContract resolve_matmul_io_contract(const IoContractRequest& request);
bool has_nondefault_v1_tile_transpose(const tt::tt_metal::Tile& tile) noexcept;

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
    bool has_unsupported_tile_metadata = false;
};

Eligibility v1_eligibility_from_call_state(
    CallSemantics call,
    IoContractStatus io_contract_status,
    bool trace_capture_active,
    bool has_bias,
    const ttnn::prim::MatmulParams& parameters,
    bool has_optional_output,
    bool input_a_sharded,
    bool input_b_sharded,
    bool output_sharded,
    bool has_unsupported_tile_metadata) noexcept;

struct TensorRequest {
    tt::tt_metal::DataType dtype;
    tt::tt_metal::Layout layout;
    tt::tt_metal::TensorMemoryLayout memory_layout;
    tt::tt_metal::BufferType buffer_type;
    std::uint32_t tile_height = 0;
    std::uint32_t tile_width = 0;

    bool operator==(const TensorRequest&) const = default;
};

struct WorkloadRequest {
    std::uint64_t logical_m = 0;
    std::uint64_t logical_k = 0;
    std::uint64_t logical_n = 0;
    std::uint64_t padded_m = 0;
    std::uint64_t padded_k = 0;
    std::uint64_t padded_n = 0;

    bool operator==(const WorkloadRequest&) const = default;
};

// Physical board, topology, firmware, session, and build identities are not
// bankable. The live grid selects the harvested exact cohort and bounds native
// candidate legality.
struct DeviceRequest {
    std::uint32_t architecture = 0;
    std::uint32_t device_count = 0;
    std::uint32_t mesh_rows = 0;
    std::uint32_t mesh_cols = 0;
    std::uint32_t compute_grid_x = 0;
    std::uint32_t compute_grid_y = 0;

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
    bool transpose_a = false;
    bool transpose_b = false;
    bool has_bias = false;
    bool has_activation = false;
    bool untilize_out = false;
    std::optional<bool> bcast_batch = std::nullopt;
    bool run_batched = false;
    std::optional<std::uint32_t> activation_op = std::nullopt;
    std::array<std::uint32_t, kMaxActivationParameters> activation_param_f32_bits{};
    std::uint8_t activation_param_count = 0;

    bool operator==(const MatmulRegistryRequest&) const = default;
};

struct RegistryRequestInspection {
    std::optional<MatmulRegistryRequest> request = std::nullopt;
    Eligibility eligibility{};
};

RegistryRequestInspection inspect_registry_request(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    bool has_bias,
    CallSemantics call_semantics,
    const ttnn::prim::MatmulParams& parameters,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    bool trace_capture_active);

std::optional<compact::KeyDescriptor> compact_registry_key(const MatmulRegistryRequest& request) noexcept;

struct Resolution {
    ResolutionReason reason = ResolutionReason::EmptyRegistry;
    std::optional<compact::ProgramConfigDescriptor> program_config = std::nullopt;
    std::optional<compact::ComputeKernelDescriptor> compute_kernel_config = std::nullopt;
    std::optional<compact::KeyDescriptor> key = std::nullopt;
};

struct DispatchResult {
    Resolution resolution;
    std::optional<ttnn::prim::MatmulParams> materialized_parameters = std::nullopt;
};

ResolutionReason preflight_v1_eligibility(const Eligibility& eligibility) noexcept;
ResolutionReason validate_v1_request_envelope(
    const MatmulRegistryRequest& request, const Eligibility& eligibility) noexcept;

// Default-on and read-only: consult the checked exact table. Every miss,
// unsupported state, or malformed artifact returns a typed fallback reason.
Resolution resolve(const MatmulRegistryRequest& request, const Eligibility& eligibility) noexcept;

DispatchResult resolve_for_dispatch(
    const std::optional<MatmulRegistryRequest>& request,
    const Eligibility& eligibility,
    const ttnn::prim::MatmulParams& legacy_parameters);

std::optional<ttnn::prim::MatmulParams> select_registry_parameters(
    const MatmulRegistryRequest& request,
    const Eligibility& eligibility,
    const ttnn::prim::MatmulParams& legacy_parameters);

std::optional<ttnn::prim::MatmulParams> materialize_parameters_for_execution(
    const Resolution& resolution, const ttnn::prim::MatmulParams& legacy_parameters);

std::optional<MatmulProgramConfig> materialize_registry_program_config(
    const compact::KeyDescriptor& key,
    const compact::ProgramConfigDescriptor& descriptor,
    std::optional<compact::ComputeKernelDescriptor> compute_kernel_config);
std::optional<DeviceComputeKernelConfig> materialize_registry_compute_kernel_config(
    const compact::ComputeKernelDescriptor& descriptor);

Resolution resolve_with_compact_table_for_testing(
    const MatmulRegistryRequest& request,
    const Eligibility& eligibility,
    const compact::TableMetadata& metadata,
    std::span<const compact::ProgramConfigExactEntry> exact_entries = {}) noexcept;

}  // namespace ttnn::operations::matmul::registry
