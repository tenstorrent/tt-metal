// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <compare>
#include <cstdint>

namespace ttnn::operations::matmul::registry::compact {

// Shared by the compact-lock validator, emitted entries, and runtime key
// construction. Changing it requires a new
// exporter/runtime contract rather than a local literal update.
inline constexpr std::uint16_t kCodegenRecipeAbi = 2;

enum class Domain : std::uint8_t { DenseMatmul = 0, DenseLinear = 1, DenseAddmm = 2 };
// Enumerator values are stable selector ABI. emit_cpp.py explicitly reorders
// canonically reviewed lock entries into this POD's defaulted runtime order.
enum class DataType : std::uint8_t { BFloat16 = 0, BFloat8B = 1, Float32 = 2 };
enum class Layout : std::uint8_t { RowMajor = 0, Tile = 1 };
enum class MemoryLayout : std::uint8_t { Interleaved = 0 };
enum class BufferType : std::uint8_t { Dram = 0, L1 = 1 };
enum class ProgramFamily : std::uint8_t { MultiCoreReuse = 0, MultiCast1D = 1, MultiCast2D = 2 };
enum class MathFidelity : std::uint8_t { LoFi = 0, HiFi2 = 1, HiFi3 = 2, HiFi4 = 3 };
enum class ThrottleLevel : std::uint8_t {
    NoThrottle = 0,
    Throttle1 = 1,
    Throttle2 = 2,
    Throttle3 = 3,
    Throttle4 = 4,
    Throttle5 = 5,
};

using Sha256 = std::array<std::uint8_t, 32>;
using RegistryEntryId = Sha256;

struct TensorDescriptor {
    BufferType buffer_type{};
    DataType dtype{};
    Layout layout{};
    MemoryLayout memory_layout{};
    std::uint16_t tile_height{};
    std::uint16_t tile_width{};

    auto operator<=>(const TensorDescriptor&) const = default;
};

struct KeyDescriptor {
    std::uint32_t architecture{};
    bool bcast_batch_present{};
    bool bcast_batch{};
    std::uint32_t board_capability_class{};
    std::uint16_t codegen_recipe_abi{};
    std::uint16_t compute_grid_x{};
    std::uint16_t compute_grid_y{};
    std::uint16_t device_count{};
    bool has_activation{};
    bool has_bias{};
    TensorDescriptor input_a{};
    TensorDescriptor input_b{};
    std::uint64_t logical_k{};
    std::uint64_t logical_m{};
    std::uint64_t logical_n{};
    std::uint16_t mesh_cols{};
    std::uint16_t mesh_rows{};
    TensorDescriptor output{};
    std::uint64_t padded_k{};
    std::uint64_t padded_m{};
    std::uint64_t padded_n{};
    bool run_batched{};
    std::uint16_t schema_version{};
    Sha256 topology_sha256{};
    bool transpose_a{};
    bool transpose_b{};
    bool untilize_out{};
    // The three public-operation domains remain disjoint exact-key axes. V1
    // admits only the exact no-bias/no-activation subset; addmm additionally
    // binds exact IEEE-754 alpha/beta spellings and admits beta +/-0 only, so
    // the otherwise unkeyed additive input is provably unused. The resolver
    // may then reuse a dense.matmul measurement for an admitted wrapper whose
    // inner matmul is identical, after preferring a domain-specific entry.
    Domain domain{};
    std::uint32_t alpha_f32_bits{};
    std::uint32_t beta_f32_bits{};

    auto operator<=>(const KeyDescriptor&) const = default;
};

struct ComputeKernelDescriptor {
    MathFidelity math_fidelity{};
    ThrottleLevel throttle_level{};
    bool math_approx_mode{};
    bool fp32_dest_acc_en{};
    bool packer_l1_acc{};
    bool dst_full_sync_en{};

    auto operator<=>(const ComputeKernelDescriptor&) const = default;
};

struct TableMetadata {
    std::uint16_t lock_schema_version{};
    std::uint16_t key_schema_version{};
    // Zero disables exact selection; schema 2 binds deterministic bank
    // evidence for the complete ProgramConfig+CKC native recipe.
    std::uint16_t exact_recipe_evidence_schema_version{};
    // Nonzero only when exact evidence explicitly authorizes eligibility-
    // proven linear/addmm aliases to the dense matmul kernel key.
    std::uint16_t matmul_kernel_equivalence_schema_version{};
    Sha256 content_sha256{};
    Sha256 semantic_source_sha256{};
};

}  // namespace ttnn::operations::matmul::registry::compact
