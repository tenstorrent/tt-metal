// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::normalization::rmsnorm_distributed_bw {

// Fused apply step of distributed RMSNorm backward:
//   dx     = gamma * dy / rms - x * scale / rms^2
//   dgamma = sum(dy * x / rms) over N, C, H   (only when gamma is set)
struct RMSNormBwApplyOperation {
    struct operation_attributes_t {
        MemoryConfig memory_config;
        DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& x;
        const Tensor& dy;
        std::optional<Tensor> gamma;
        const Tensor& inv_rms;
        const Tensor& d;
    };

    using spec_return_value_t = std::vector<std::optional<tt::tt_metal::TensorSpec>>;
    using tensor_return_value_t = std::vector<std::optional<Tensor>>;

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& outputs);

        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& outputs,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

// Occupancy used by the program hash and the factory so they cannot drift.
// num_rows is a runtime arg; two shapes that fill the same grid at the same Wt share a program.
struct ApplyOccupancy {
    uint32_t Wt = 0;
    uint32_t num_rows = 0;
    uint32_t num_cores = 0;
    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    // Grid rows the cores span, i.e. the number of row leaders in the dgamma reduction tree.
    uint32_t n_rows_used = 0;
};

ApplyOccupancy compute_apply_occupancy(const Tensor& x);

}  // namespace ttnn::operations::normalization::rmsnorm_distributed_bw

namespace ttnn::prim {

std::vector<std::optional<Tensor>> rmsnorm_bw_apply(
    const Tensor& x,
    const Tensor& dy,
    const std::optional<Tensor>& gamma,
    const Tensor& inv_rms,
    const Tensor& d,
    const MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
