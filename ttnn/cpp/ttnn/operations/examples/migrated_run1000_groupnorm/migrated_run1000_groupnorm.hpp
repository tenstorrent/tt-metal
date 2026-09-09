// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Native TTNN migration of the registry-model Python operation
// `groupnorm_sc_N_1_HW_C` (regime `cluster_parallel_two_pass`).
//
// Everything the Python entry point and its ProgramDescriptor builder did on the
// host is translated here into typed C++:
//
//   * `_validate_args()`  -> `validate_arguments()`      (ValueError)
//   * Python `validate`  -> `validate_support()`        (UnsupportedAxisValue /
//                                                         ExcludedCell)
//   * `create_program_descriptor()` -> `build_plan()` (all planner branches:
//     cluster geometry, mask spans, L1 budget, regime selection, residency
//     fast path, block-knob snapping, work split, Mcast1D argument derivation)
//     plus `build_descriptor()` (CBs, kernels, compile/runtime args, semaphores)
//   * `ttnn.allocate_tensor_on_device(...)` -> `compute_output_specs` /
//     `create_output_tensors()`
//   * `ttnn.generic_op(...)` -> `device_operation::launch<...>()` of the
//     distinct named device operation declared below.
//
// The Python-visible exception identities are preserved: the host code throws
// the `HostError` carrier and the nanobind boundary re-raises the very same
// Python classes (`ValueError`, `NotImplementedError`, `ZeroDivisionError`,
// `ttnn.operations._op_contract.UnsupportedAxisValue`, and
// `ttnn.operations._op_contract.ExcludedCell`).

#pragma once

#include <array>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::migration::generated_migrated_run1000_groupnorm {

// ---------------------------------------------------------------------------
// Python exception identities reachable from the source operation
// ---------------------------------------------------------------------------
enum class PyErrKind : std::uint8_t {
    ValueError,            // builtins.ValueError            (_validate_args)
    NotImplementedError,   // builtins.NotImplementedError   (L1 fit failure)
    ZeroDivisionError,     // builtins.ZeroDivisionError     (degenerate knobs)
    UnsupportedAxisValue,  // ttnn.operations._op_contract.UnsupportedAxisValue
    ExcludedCell,          // ttnn.operations._op_contract.ExcludedCell
};

// Carrier for a host refusal. Not a new exception identity: the nanobind
// boundary translates it back into the exact Python class the source raised.
class HostError : public std::exception {
public:
    HostError(PyErrKind kind, std::string message) : kind_(kind), message_(std::move(message)) {}

    PyErrKind kind() const noexcept { return kind_; }
    const char* what() const noexcept override { return message_.c_str(); }

private:
    PyErrKind kind_;
    std::string message_;
};

// ---------------------------------------------------------------------------
// `_DEFAULT_COMPUTE_CONFIG` / `_compute_config_fields()` projection
// ---------------------------------------------------------------------------
// Phase-0 defaults, preserved exactly when the caller passes no config.
struct ComputeConfigFields {
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    bool fp32_dest_acc_en = true;
    bool math_approx_mode = false;
    bool dst_full_sync_en = false;
};

// ---------------------------------------------------------------------------
// Per-core runtime-argument plan (address-free: buffers are bound separately)
// ---------------------------------------------------------------------------
struct CoreArgs {
    tt::tt_metal::CoreCoord core{0, 0};
    std::uint32_t unit_start = 0;
    std::uint32_t units = 0;
    std::uint32_t hw_tile_start = 0;
    std::uint32_t is_sender = 0;
    std::uint32_t gather_slot = 0;
    std::uint32_t owns_hw_tail = 0;
    std::array<std::uint32_t, 4> mcast_rt{0u, 0u, 0u, 0u};
};

// ---------------------------------------------------------------------------
// The whole host plan. Structural + scalar only; contains no buffer address.
// ---------------------------------------------------------------------------
struct Plan {
    // --- tensor geometry ---
    std::uint32_t N = 0;
    std::uint32_t HW = 0;
    std::uint32_t C = 0;
    std::uint32_t num_groups = 0;

    // --- channel-cluster construction ---
    std::uint32_t Cg = 0;
    std::uint32_t cluster_channels = 0;
    std::uint32_t groups_per_cluster = 0;
    std::uint32_t cluster_c_tiles = 0;
    std::uint32_t num_clusters = 0;
    std::uint32_t num_mask_tiles = 0;
    std::uint32_t max_span = 0;

    std::uint32_t tensor_hw_tiles = 0;
    std::uint32_t tensor_c_tiles = 0;
    std::uint32_t hw_tail = 0;
    std::uint32_t c_tail = 0;

    // --- formats ---
    bool input_is_rm = false;
    std::uint32_t in_elem = 0;
    std::uint32_t tile_in = 0;
    std::uint32_t tile_out = 0;
    std::uint32_t tile_f32 = 0;
    std::uint32_t tile_bf16 = 0;
    tt::tt_metal::DataType input_dtype = tt::tt_metal::DataType::BFLOAT16;
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::BFLOAT16;

    // --- affine operands ---
    bool has_gamma = false;
    bool has_beta = false;
    bool affine_is_rm = false;
    std::uint32_t affine_elem = 0;
    std::uint32_t affine_page_bytes = 0;
    std::uint32_t affine_scratch_bytes = 0;

    // --- CB depths / block knobs ---
    std::uint32_t input_depth = 0;
    std::uint32_t output_depth = 0;
    std::uint32_t rm_depth = 0;
    std::uint32_t block_hw_tiles = 0;
    std::uint32_t num_hw_blocks = 0;
    std::uint32_t block_tiles = 0;
    std::uint32_t input_cb_pages = 0;
    std::uint32_t per_core_hw_tiles = 0;
    bool resident = false;

    // --- regime ---
    std::uint32_t regime = 0;
    std::uint32_t hw_split_factor = 1;
    std::uint32_t units_total = 0;

    // --- compute-kernel scalars ---
    std::uint32_t inv_n_g_bits = 0;
    std::uint32_t eps_bits = 0;

    // --- work distribution ---
    tt::tt_metal::CoreRangeSet all_cores;
    std::vector<CoreArgs> core_args;

    // --- Mcast1D derived state ---
    std::array<std::uint32_t, 6> mcast_ct{0u, 0u, 0xFFFFFFFFu, 0u, 0u, 0u};
    std::vector<tt::tt_metal::SemaphoreDescriptor> semaphores;

    // --- compute config ---
    ComputeConfigFields compute;
};

// ---------------------------------------------------------------------------
// The distinct named device operation
// ---------------------------------------------------------------------------
struct MigratedRun1000GroupNormDeviceOperation {
    struct operation_attributes_t {
        Plan plan;
        double eps = 1e-5;

        static constexpr auto attribute_names = std::forward_as_tuple(
            "num_groups",
            "eps",
            "regime",
            "hw_split_factor",
            "block_hw_tiles",
            "num_hw_blocks",
            "resident",
            "cluster_c_tiles",
            "groups_per_cluster",
            "num_clusters",
            "units_total");
        std::tuple<
            std::uint32_t,
            double,
            std::uint32_t,
            std::uint32_t,
            std::uint32_t,
            std::uint32_t,
            bool,
            std::uint32_t,
            std::uint32_t,
            std::uint32_t,
            std::uint32_t>
        attribute_values() const {
            return std::make_tuple(
                plan.num_groups,
                eps,
                plan.regime,
                plan.hw_split_factor,
                plan.block_hw_tiles,
                plan.num_hw_blocks,
                plan.resident,
                plan.cluster_c_tiles,
                plan.groups_per_cluster,
                plan.num_clusters,
                plan.units_total);
        }
    };

    struct tensor_args_t {
        Tensor input_tensor;
        std::optional<Tensor> gamma;
        std::optional<Tensor> beta;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    // Descriptor program factory. `override_runtime_arguments` re-derives the
    // whole per-dispatch state (every runtime arg, including every buffer base
    // address) on a program-cache hit, so the framework needs neither address
    // inference nor the legacy dynamic-arg hook.
    struct ClusterParallelTwoPass {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
    };

    using program_factory_t = std::variant<ClusterParallelTwoPass>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

// ---------------------------------------------------------------------------
// Host entry points (all planning/validation is native)
// ---------------------------------------------------------------------------

// `_validate_args()` -- argument validation, ValueError (NOT NotImplementedError).
void validate_arguments(
    const Tensor& input_tensor,
    std::int64_t num_groups,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta);

// Python `validate` -- INPUT_TAGGERS / SUPPORTED / EXCLUSIONS support contract.
void validate_support(
    const Tensor& input_tensor,
    std::int64_t num_groups,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta);

// `create_program_descriptor()` planner half: every derived scalar, the regime
// selection, the residency predicate, the block-knob snapping and the work
// split. May raise NotImplementedError (L1 fit) or ZeroDivisionError.
Plan build_plan(
    const Tensor& input_tensor,
    tt::tt_metal::DataType output_dtype,
    std::int64_t num_groups,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta,
    double eps,
    const ComputeConfigFields& compute);

// The public operation body: validate, plan, allocate, launch.
Tensor invoke(
    const Tensor& input_tensor,
    std::int64_t num_groups,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta,
    double eps,
    const ComputeConfigFields& compute);

}  // namespace ttnn::migration::generated_migrated_run1000_groupnorm
