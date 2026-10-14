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
        // Row of the kv output that selected row 0 is written to.
        uint32_t output_row_offset = 0;
        // Candidate source mode when > 0: also select the ``candidate_topk_blocks`` best blocks of
        // ``candidate_block_size`` keys (each scored by its best key, the newest block always kept).
        uint32_t candidate_topk_blocks = 0;
        uint32_t candidate_block_size = 0;
        // Key split when > 1: this device scores part ``part_index_tensor[0]`` of ``num_parts``
        // contiguous ranges of page blocks and outputs its local top-k candidates instead of
        // gathering rows (see ``merge_select_kv``).
        uint32_t num_parts = 1;
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
        // Preallocated kv output ``[1, Hkv, rows, Dh]``; the selected rows land at
        // ``[output_row_offset, output_row_offset + k)``.
        std::optional<Tensor> output_tensor;
        // Written into ``kv_cache`` at logical row ``new_kv_row_index[0]`` before the gather.
        std::optional<Tensor> new_kv_row;
        std::optional<Tensor> new_kv_row_index;
        // ``[1]`` INT32 number of selectable keys; replaces ``(cur_pos + 1) / 4`` when given.
        std::optional<Tensor> num_keys_tensor;
        // Preallocated candidate source outputs: ``[1, candidate_topk_blocks]`` and ``[1]`` INT32.
        std::optional<Tensor> candidate_blocks_output;
        std::optional<Tensor> pool_count_output;
        // ``[1]`` INT32 part this device scores, with ``num_parts`` > 1.
        std::optional<Tensor> part_index_tensor;
        // ``[1, max_blocks_per_user * block_size]`` INT32 ``kv_cache`` row of each key, replacing the
        // page-table mapping for the gather (a candidate pool's rows in the source's cache).
        std::optional<Tensor> row_table;
    };

    // [0] selected kv rows (with num_parts > 1: the local candidates, ``[1, 1, 1, 2k + 16]`` INT32:
    // k kv_cache rows, their k order keys, then the number selected), [1] index scores; in candidate
    // source mode also [2] candidate blocks, [3] pool count.
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
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& output_tensor = std::nullopt,
    uint32_t output_row_offset = 0,
    const std::optional<ttnn::Tensor>& new_kv_row = std::nullopt,
    const std::optional<ttnn::Tensor>& new_kv_row_index = std::nullopt,
    const std::optional<ttnn::Tensor>& num_keys_tensor = std::nullopt,
    uint32_t candidate_topk_blocks = 0,
    uint32_t candidate_block_size = 0,
    const std::optional<ttnn::Tensor>& candidate_blocks_output = std::nullopt,
    const std::optional<ttnn::Tensor>& pool_count_output = std::nullopt,
    uint32_t num_parts = 1,
    const std::optional<ttnn::Tensor>& part_index_tensor = std::nullopt,
    const std::optional<ttnn::Tensor>& row_table = std::nullopt);

}  // namespace ttnn::prim
