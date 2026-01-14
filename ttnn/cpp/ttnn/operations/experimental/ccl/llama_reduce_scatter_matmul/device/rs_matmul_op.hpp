// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

#include "ttnn/operation.hpp"

#include <optional>
#include <vector>
#include <algorithm>

/* Fusion includes */

#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_device_operation.hpp"

namespace ttnn::operations::experimental::ccl {

struct Matmul_RS {
    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    struct matmul_tensor_args_t {
        const Tensor input_tensor;
        const Tensor weight_tensor;
    };
    struct tensor_args_t {
        LlamaReduceScatterDeviceOperation::tensor_args_t rs;
        matmul_tensor_args_t matmul;
        std::vector<Tensor> matmul_output_tensors;
        const std::optional<const ttnn::Tensor> second_weight_tensor;
    };
    struct operation_attributes_t {
        LlamaReduceScatterDeviceOperation rs;
        LlamaReduceScatterDeviceOperation::operation_attributes_t rs_op;
        matmul::MatmulDeviceOperation::operation_attributes_t matmul;
        using matmul_device_t = matmul::MatmulDeviceOperation;
    };
    struct Matmul_RS_PF {
        // Shared variables are the variables that are shared between the create and override_runtime_arguments methods
        struct shared_variables_t {
            LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::shared_variables_t rs_shared_vars;
            matmul::program::matmul_mcast_1d_common_override_variables_t matmul_shared_vars;
        };
        using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

        static cached_mesh_workload_t create_mesh_workload(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            std::vector<Tensor>& tensor_return_value);

        static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinate& mesh_coordinate,
            const tensor_args_t& tensor_args,
            std::vector<Tensor>& tensor_return_value);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            std::vector<Tensor>& tensor_return_value);
    };
    using program_factory_t = std::variant<Matmul_RS_PF>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::ccl

namespace ttnn::prim {

ttnn::operations::experimental::ccl::Matmul_RS::tensor_return_value_t llama_rs_matmul(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::Tensor>& rs_tensor,
    ttnn::Tensor& intermediate_packet_buffer,
    int32_t dim,
    const GlobalSemaphore& semaphore,
    uint32_t cluster_axis,
    uint32_t ring_devices,
    uint32_t num_links,
    const tt::tt_metal::SubDeviceId& subdevice_id,
    const std::optional<ttnn::MemoryConfig>& memory_config_rs = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config_mm = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<const GlobalCircularBuffer>& global_cb = std::nullopt,
    std::optional<const ttnn::CoreGrid> core_grid = std::nullopt,
    bool transpose_a = false,
    bool transpose_b = false,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
    const std::optional<const std::string>& activation = std::nullopt,
    const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear,
    bool use_noc1_only = false,
    const std::optional<const ttnn::Tensor>& second_weight_tensor = std::nullopt);

}  // namespace ttnn::prim
