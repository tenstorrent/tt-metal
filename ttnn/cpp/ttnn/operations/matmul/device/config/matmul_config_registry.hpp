// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tile.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
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

enum class ResolutionReason {
    Disabled,
    IneligibleOperationDomain,
    MalformedOperationSemantics,
    InconsistentIoContract,
    ExplicitOverride,
    UnsupportedSemantics,
    IncompleteRequest,
    InconsistentRequest,
    EmptyRegistry,
    CertifiedMatch,
    Count,
};

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
    std::uint32_t architecture;
    std::uint32_t device_count;
    std::uint32_t mesh_rows;
    std::uint32_t mesh_cols;
    std::uint32_t compute_grid_x;
    std::uint32_t compute_grid_y;

    bool operator==(const DeviceRequest&) const = default;
};

struct MatmulRegistryRequest {
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
    // Exact IEEE-754 binary32 spellings, in parameter order.
    std::vector<std::uint32_t> activation_param_f32_bits;

    bool operator==(const MatmulRegistryRequest&) const = default;
};

struct Recipe {
    MatmulProgramConfig program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    // Duplicated native state is selected explicitly rather than inferred from
    // one program-config alternative at the call site.
    bool untilize_out = false;
};

struct Resolution {
    ResolutionReason reason = ResolutionReason::Disabled;
    std::optional<Recipe> recipe = std::nullopt;
};

enum class ExecutionAction { Fallback, ObserveOnly, ApplyRecipe };

// A certified Shadow hit is observable but never mutates execution parameters.
ExecutionAction execution_action(Mode mode, const Resolution& resolution) noexcept;

// Every recipe carries one effective untilize_out value. It must agree with the
// duplicated field in the 1D config, and must be false for all other families.
bool has_consistent_untilize_out(const Recipe& recipe) noexcept;

// Snapshots CONFIG on first use. Subsequent configuration mutation cannot alter
// behavior during the process lifetime.
Mode current_mode() noexcept;

// Test only: callers must ensure no concurrent matmul dispatch is in flight.
void reset_startup_mode_for_testing() noexcept;

// Device-free admission and empty-table lookup. A future table implementation
// may return a Recipe only after exactly matching the complete typed request key.
Resolution resolve(Mode mode, const MatmulRegistryRequest& request, const Eligibility& eligibility) noexcept;

struct DomainStatsSnapshot {
    std::uint64_t resolution_attempts = 0;
    std::uint64_t certified_hits = 0;
    std::uint64_t shadow_would_hits = 0;
    std::uint64_t applied_hits = 0;
    std::uint64_t fallbacks = 0;
    std::array<std::uint64_t, kResolutionReasonCount> reasons{};
};

struct StatsSnapshot {
    bool mode_is_frozen = false;
    Mode frozen_mode = Mode::Off;
    std::array<DomainStatsSnapshot, kOperationDomainCount> domains{};
};

// Telemetry cardinality is bounded by fixed domain and reason enums. Request
// dimensions and identifiers are deliberately never retained.
void record_resolution(
    Mode mode, OperationDomain domain, const Resolution& resolution, ExecutionAction action) noexcept;
StatsSnapshot stats_snapshot() noexcept;

// Test only: callers must ensure no concurrent recording is in flight.
void reset_stats_for_testing() noexcept;

}  // namespace ttnn::operations::matmul::registry
