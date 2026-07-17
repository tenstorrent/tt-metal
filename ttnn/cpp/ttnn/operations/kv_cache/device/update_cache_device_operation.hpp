// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/device_operation.hpp"
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include "ttnn/distributed/types.hpp"
#include "update_cache_device_operation_types.hpp"
#include "fill_cache_multi_core_program_factory.hpp"
#include "update_cache_multi_core_program_factory.hpp"

namespace ttnn::prim {

struct UpdateKVCacheOperation {
    using operation_attributes_t = KvCacheParams;
    using tensor_args_t = KvCacheInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<UpdateCacheMultiCoreProgramFactory, FillCacheMultiCoreProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static tt::tt_metal::operation::Hash compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    // Cache-hit re-apply of all per-dispatch state (per-core args + tensor-backed CB/buffer
    // addresses), since compute_program_hash excludes batch_idx/update_idx/batch_offset. See the .cpp.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

Tensor update_cache(
    const Tensor& cache,
    const Tensor& input,
    uint32_t batch_idx,
    uint32_t update_index,
    uint32_t batch_offset,
    UpdateCacheOpType op_type,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::prim
