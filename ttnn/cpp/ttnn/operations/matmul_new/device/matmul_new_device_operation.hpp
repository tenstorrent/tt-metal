// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/host_api.hpp>

#include "ttnn/operations/matmul_new/device/factory/multicore_descriptor.hpp"
#include "ttnn/operations/matmul_new/device/factory/reuse_optimized_descriptor.hpp"
#include "ttnn/operations/matmul_new/device/factory/reuse_mcast_1d_descriptor.hpp"
#include "ttnn/operations/matmul_new/device/factory/reuse_mcast_2d_descriptor.hpp"
#include "ttnn/operations/matmul_new/device/factory/dram_sharded_descriptor.hpp"
#include "ttnn/operations/matmul_new/device/factory/batched_hs_dram_sharded_descriptor.hpp"

namespace ttnn::prim {

// ---------------------------------------------------------------
// MatmulNewDeviceOperation -- ProgramDescriptor variant of matmul.
//
// Functionally identical to MatmulDeviceOperation but ALL program
// factories use ProgramDescriptor for construction.
//
// All 6 factories are in the variant directly:
//   - MultiCoreDescriptorFactory (MeshWorkloadFactoryConcept)
//   - 5 ProgramDescriptorFactoryConcept factories (wrapped by the
//     framework's DescriptorMeshWorkloadFactoryAdapter automatically)
// ---------------------------------------------------------------
struct MatmulNewDeviceOperation {
    using operation_attributes_t = MatmulParams;
    using tensor_args_t = MatmulInputs;
    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    using program_factory_t = std::variant<
        matmul_new_detail::MultiCoreDescriptorFactory,
        matmul_new_detail::ReuseOptimizedDescriptorFactory,
        matmul_new_detail::ReuseMcast2DDescriptorFactory,
        matmul_new_detail::ReuseMcast1DDescriptorFactory,
        matmul_new_detail::DRAMShardedDescriptorFactory,
        matmul_new_detail::BatchedHSDRAMShardedDescriptorFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    // Optional: custom hash mirrors original matmul's fast attribute-based hash.
    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};

}  // namespace ttnn::prim

namespace ttnn::prim {

MatmulNewDeviceOperation::tensor_return_value_t matmul_new(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<Tensor>& bias = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const MatmulParams& attributes = MatmulParams());

}  // namespace ttnn::prim
