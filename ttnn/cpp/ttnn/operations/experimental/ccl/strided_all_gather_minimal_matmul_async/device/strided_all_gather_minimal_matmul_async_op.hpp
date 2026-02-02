// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/global_semaphore.hpp"

#include "ttnn/operation.hpp"

#include <optional>
#include <vector>
#include <algorithm>

/* Fusion includes */
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

#include "strided_all_gather_minimal_matmul_async_device_operation_types.hpp"
#include "strided_all_gather_minimal_matmul_async_program.hpp"

namespace ttnn::experimental::prim {

struct StridedAllGatherMinimalMatmulAsync {
    using operation_attributes_t = StridedAllGatherMinimalMatmulAsyncParams;
    using tensor_args_t = StridedAllGatherMinimalMatmulAsyncInputs;
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    using program_factory_t = std::variant<StridedAllGatherMinimalMatmulAsyncProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::tt_metal::operation::Hash compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> strided_all_gather_minimal_matmul_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    CoreCoord strided_all_gather_core_grid_offset,
    uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config_ag,
    ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    const std::optional<const Tensor>& bias,
    const std::optional<MemoryConfig>& memory_config_mm,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
    std::optional<const ttnn::experimental::prim::MinimalMatmulConfig> config,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    std::optional<bool> read_local_slice_from_input);

}  // namespace ttnn::prim
