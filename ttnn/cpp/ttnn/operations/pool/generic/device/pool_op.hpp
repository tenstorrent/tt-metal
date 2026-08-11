// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

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

    using spec_return_value_t = tt::tt_metal::TensorSpec;
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
};

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
