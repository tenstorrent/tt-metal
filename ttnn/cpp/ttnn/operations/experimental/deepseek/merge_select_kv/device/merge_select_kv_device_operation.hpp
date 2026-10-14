// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::merge_select_kv {

struct MergeSelectKvDeviceOperation {
    struct operation_attributes_t {
        uint32_t k = 0;
        // Row of the output that selected row 0 is written to.
        uint32_t output_row_offset = 0;
        MemoryConfig output_mem_config;
    };

    struct tensor_args_t {
        const Tensor& candidates;
        const Tensor& kv_cache;
        std::optional<Tensor> output_tensor;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

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

}  // namespace ttnn::operations::experimental::deepseek::merge_select_kv

namespace ttnn::prim {

ttnn::Tensor merge_select_kv(
    const ttnn::Tensor& candidates,
    const ttnn::Tensor& kv_cache,
    uint32_t k,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::Tensor>& output_tensor = std::nullopt,
    uint32_t output_row_offset = 0);

}  // namespace ttnn::prim
