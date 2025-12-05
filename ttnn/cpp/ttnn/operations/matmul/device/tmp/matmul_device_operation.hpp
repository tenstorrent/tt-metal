// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/matmul/device/tmp/matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/tmp/factory/matmul_multicore_program_factory.hpp"
// #include "ttnn/operations/matmul/device/tmp/factory/matmul_multicore_reuse_program_factory.hpp"
#include "ttnn/operations/matmul/device/tmp/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp"
#include "ttnn/operations/matmul/device/tmp/factory/matmul_multicore_reuse_mcast_2d_program_factory.hpp"
#include "ttnn/operations/matmul/device/tmp/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.hpp"
#include "ttnn/operations/matmul/device/tmp/factory/matmul_multicore_reuse_optimized_program_factory.hpp"

namespace ttnn::operations::matmul {

struct MatmulDeviceOperation {
    using operation_attributes_t = matmul::operation_attributes_t;
    using tensor_args_t = matmul::tensor_args_t;
    using spec_return_value_t = matmul::spec_return_value_t;
    using tensor_return_value_t = matmul::tensor_return_value_t;

    using program_factory_t = std::variant<
        program::MatmulMeshWorkloadMultiCoreFactory,
        program::MatmulMeshWorkloadMultiCoreReuseOptimizedProgramFactory,
        program::MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory,
        program::MatmulMeshWorkloadMultiCoreReuseMcast2DProgramFactory,
        program::MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<Tensor>& bias = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<ttnn::operations::matmul::config::MatmulProgramConfig>& program_config = std::nullopt,
        const std::optional<bool>& bcast_batch = std::nullopt,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        const std::optional<DataType>& output_dtype = std::nullopt,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
        const std::optional<CoreCoord>& user_core_coord = std::nullopt,
        const std::optional<ttnn::operations::unary::UnaryWithParam>& user_fused_activation = std::nullopt,
        bool transpose_a = false,
        bool transpose_b = false,
        const std::optional<tt::tt_metal::Tile>& output_tile = std::nullopt,
        const std::optional<GlobalCircularBuffer>& global_cb = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt);
};

}  // namespace ttnn::operations::matmul

namespace ttnn::prim {
constexpr auto matmul =
    ttnn::register_operation<"ttnn::prim::matmul", ttnn::operations::matmul::MatmulDeviceOperation>();
}  // namespace ttnn::prim
