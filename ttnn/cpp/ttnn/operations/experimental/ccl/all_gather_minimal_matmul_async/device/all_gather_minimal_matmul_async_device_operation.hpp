// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

#include "all_gather_minimal_matmul_async_device_operation_types.hpp"
#include "all_gather_minimal_matmul_async_program_factory.hpp"

namespace ttnn::experimental::prim {

struct AllGatherMinimalMatmulAsyncOp {
    using operation_attributes_t = AllGatherMinimalMatmulAsyncParams;
    using tensor_args_t = AllGatherMinimalMatmulAsyncInputs;
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    using program_factory_t = std::variant<AllGatherMinimalMatmulAsyncProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::tt_metal::operation::Hash compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::Tensor all_gather_minimal_matmul_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& bias_tensor,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
    const std::optional<const experimental::prim::AllGatherMinimalMatmulAsyncConfig>& config,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const ttnn::ccl::Topology topology,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    uint32_t num_links,
    std::optional<uint32_t> cluster_axis,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const bool force_transpose,
    uint32_t num_workers_per_link,
    uint32_t num_buffers_per_channel);

}  // namespace ttnn::prim
