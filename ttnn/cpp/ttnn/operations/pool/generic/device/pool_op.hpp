// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include <utility>

namespace ttnn::operations::pool {

// Generic pool uop -- called from the macro-ops
struct Pool2D {
    struct operation_attributes_t {
        sliding_window::SlidingWindowConfig sliding_window_config_{};
        Pool2DType pool_type_{};
        DataType output_dtype_{};
        Layout output_layout_{};
        MemoryConfig memory_config_;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config_;
        bool count_include_pad_{};
        std::optional<int32_t> divisor_override_;
        bool return_indices_{};
        uint32_t memory_used{};
        bool config_tensor_in_dram{};
    };

    struct tensor_args_t {
        const Tensor& input_tensor_;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = std::vector<Tensor>;

    struct MultiCore {
        // Builds the entire workload in one call (cache miss):
        //   1. Uploads the halo lookup table (and, for avg-pool variants that
        //      need it, the per-stick scalar config tensor) and parks the
        //      backing MeshBuffers in the descriptor's `buffers` vector so
        //      they outlive the cached workload.
        //   2. Loops `tensor_coords` and pushes a ProgramDescriptor per coord
        //      into `programs`.
        static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
            const operation_attributes_t& op_attr,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensors,
            const ttnn::MeshCoordinateRangeSet& tensor_coords);
    };

    using program_factory_t = std::variant<MultiCore>;
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);

    // On a program-cache hit the descriptor fast-path skips create_workload_descriptor() and instead calls
    // this to re-apply per-dispatch runtime args. To stay correct under any future hash relaxation we
    // re-apply the COMPLETE per-core reader0/reader1/compute runtime-arg state, derived from the same helper
    // (compute_pool_per_core_runtime_args) the cache-miss build uses (single source of truth).
    //
    // Rationale (verified against pool_multi_core_program_factory.cpp):
    //   * Every per-dispatch tensor address rides on a sharded CBDescriptor.buffer binding that the
    //     framework patches automatically on a cache hit: input via raw_in_cb (input.buffer()),
    //     output via out_cb (outputs[0].buffer()), and the optional index output via out_idx_cb
    //     (outputs[1].buffer()). No tensor address appears in the kernel runtime args.
    //   * The only kernel RUNTIME args (reader0/reader1/compute) are {out_nhw_this_core,
    //     core_nhw_index} plus, when return_indices, {start_row, start_col}. All are derived purely
    //     from the SlidingWindowConfig / sharding, and every attribute they depend on is covered by
    //     compute_program_hash -- so the work-split (and hence every per-core arg value and the set of
    //     cores) is identical across all dispatches that share a cache entry. There is no shape
    //     variation within a cache entry as there is for unary / binary_ng (whose hash omits
    //     padded_shape); we still re-apply the full per-core state here for safety / robustness.
    //   * The auxiliary reader_indices and scalar_config buffer addresses/page_sizes are baked into
    //     COMPILE-TIME args (not runtime args). Those buffers are allocated once in
    //     create_workload_descriptor and parked in WorkloadDescriptor::buffers, so they stay alive and
    //     keep the same address for the cached workload's lifetime (not re-created on a hit) -- the
    //     baked compile-time values remain valid.
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&,
        const std::optional<ttnn::MeshCoordinate>& = std::nullopt);
};

// Single source of truth for the per-core reader/compute RUNTIME args of the pool program.
// Both create_workload_descriptor() (cache miss, via pool2d_multi_core_sharded_with_halo_v2_impl_new)
// and get_dynamic_runtime_args() (cache hit re-apply) derive their per-core args from
// compute_pool_per_core_runtime_args(), so the values written can never drift between the two paths.
// `args[i]` is the full runtime-arg vector for `cores[i]`: {out_nhw_this_core, core_nhw_index} and,
// when return_indices is set, {start_row, start_col} appended. None of these are tensor addresses
// (addresses ride on sharded CBDescriptor.buffer bindings patched by the framework), so every slot is
// re-applied verbatim on a cache hit.
struct PoolPerCoreRuntimeArgs {
    std::vector<tt::tt_metal::CoreCoord> cores;
    std::vector<tt::tt_metal::KernelDescriptor::CoreRuntimeArgs> args;  // parallel to `cores`
};

PoolPerCoreRuntimeArgs compute_pool_per_core_runtime_args(
    const Pool2D::operation_attributes_t& op_attr,
    const Pool2D::tensor_args_t& tensor_args,
    const Pool2D::tensor_return_value_t& outputs);

}  // namespace ttnn::operations::pool

namespace ttnn::prim {
std::vector<ttnn::Tensor> pool2d(
    const Tensor& input_tensor,
    const ttnn::operations::sliding_window::SlidingWindowConfig& sliding_window_config,
    ttnn::operations::pool::Pool2DType pool_type,
    DataType output_dtype,
    Layout output_layout,
    MemoryConfig memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    bool count_include_pad,
    std::optional<int32_t> divisor_override,
    bool return_indices,
    uint32_t memory_used,
    bool config_tensor_in_dram);
}  // namespace ttnn::prim
