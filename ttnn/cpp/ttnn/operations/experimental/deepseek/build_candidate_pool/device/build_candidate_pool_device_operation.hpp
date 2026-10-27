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
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::build_candidate_pool {

struct BuildCandidatePoolDeviceOperation {
    struct operation_attributes_t {
        // Keys per candidate block.
        uint32_t candidate_block_size = 0;
    };

    struct tensor_args_t {
        const Tensor& candidate_blocks;
        const Tensor& pool_count;
        const Tensor& key_cache;
        const Tensor& page_table_tensor;
        std::optional<Tensor> pool_keys_output;
        std::optional<Tensor> pool_rows_output;
        // ``[1]`` INT32 row of a ``[num_sessions, max_blocks_per_user]`` page table to read.
        std::optional<Tensor> page_table_row_tensor;
    };

    // [0] pool keys, [1] pool rows (the kv cache row of each pool key).
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

}  // namespace ttnn::operations::experimental::deepseek::build_candidate_pool

namespace ttnn::prim {

std::vector<ttnn::Tensor> build_candidate_pool(
    const ttnn::Tensor& candidate_blocks,
    const ttnn::Tensor& pool_count,
    const ttnn::Tensor& key_cache,
    const ttnn::Tensor& page_table_tensor,
    uint32_t candidate_block_size,
    const std::optional<ttnn::Tensor>& pool_keys_output = std::nullopt,
    const std::optional<ttnn::Tensor>& pool_rows_output = std::nullopt,
    const std::optional<ttnn::Tensor>& page_table_row_tensor = std::nullopt);

}  // namespace ttnn::prim
