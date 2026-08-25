// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tile.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"

namespace ttnn::operations::matmul::registry {

enum class Mode { Off, Shadow, On };

// The shared bound_matmul helper serves three public operations. The domain is
// explicit so a recipe certified for one operation can never match another.
enum class OperationDomain { DenseMatmul, Linear, Addmm, IneligibleSharedCaller };

struct CallSemantics {
    OperationDomain domain = OperationDomain::IneligibleSharedCaller;
    // Exact IEEE-754 binary32 spellings are part of addmm semantics. They stay
    // absent for every other domain and avoid lossy decimal normalization.
    std::optional<std::uint32_t> alpha_f32_bits = std::nullopt;
    std::optional<std::uint32_t> beta_f32_bits = std::nullopt;
};

enum class ResolutionReason {
    Disabled,
    IneligibleOperationDomain,
    MalformedOperationSemantics,
    InconsistentIoContract,
    ExplicitOverride,
    UnsupportedSemantics,
    EmptyRegistry,
    CertifiedMatch,
};

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
    bool input_a_sharded = false;
    bool input_b_sharded = false;
    bool input_b_batched = false;
    bool transpose_a = false;
    bool transpose_b = false;
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

// Startup-frozen rollout control is added with the generated table. The first
// plumbing change is deliberately and unconditionally Off in production.
inline constexpr Mode current_mode() noexcept { return Mode::Off; }

// Allocation-free/device-free admission and empty-table lookup. A future table
// implementation may return a Recipe only after constructing and exactly matching
// the complete typed request key.
Resolution resolve(Mode mode, const Eligibility& eligibility) noexcept;

}  // namespace ttnn::operations::matmul::registry
