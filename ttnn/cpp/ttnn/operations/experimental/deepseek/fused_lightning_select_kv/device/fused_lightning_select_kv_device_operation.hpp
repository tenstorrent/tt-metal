// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::fused_lightning_select_kv {

struct FusedLightningSelectKvDeviceOperation {
    struct operation_attributes_t {
        uint32_t k = 0;
        MemoryConfig output_mem_config;
        ttnn::DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& query;
        const Tensor& key_cache;
        const Tensor& head_weights;
        const Tensor& kv_cache;
        const Tensor& page_table_tensor;
        const Tensor& cur_pos_tensor;
        std::optional<Tensor> valid_length_tensor;
    };

    // [0] selected kv rows, [1] index scores.
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    struct ProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek::fused_lightning_select_kv

namespace ttnn::prim {

std::vector<ttnn::Tensor> fused_lightning_select_kv(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key_cache,
    const ttnn::Tensor& head_weights,
    const ttnn::Tensor& kv_cache,
    const ttnn::Tensor& page_table_tensor,
    const ttnn::Tensor& cur_pos_tensor,
    uint32_t k,
    const std::optional<ttnn::Tensor>& valid_length_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config);

}  // namespace ttnn::prim
