// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "tt_metal/common/constants.hpp"

namespace ttnn::operations::experimental::transformer {

    operation::ProgramWithCallbacks multi_core_create_qkv_heads_sharded(const Tensor &input_tensor_qkv, const uint32_t num_q_heads, const uint32_t num_kv_heads, const uint32_t head_dim, const bool transpose_k_heads, std::vector<Tensor>& output, CoreCoord compute_with_storage_grid_size);

    struct CreateQKVHeadsDeviceOperation {
        uint32_t num_q_heads;
        uint32_t num_kv_heads;
        uint32_t head_dim;
        bool transpose_k_heads;
        MemoryConfig output_mem_config;
        void validate(const std::vector<Tensor>& input_tensors) const;
        std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
        std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
        operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    };
}  // namespace ttnn::operations::experimental::transformer
