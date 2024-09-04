// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "tt_metal/common/constants.hpp"

namespace ttnn::operations::experimental::transformer {

    operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_decode(const Tensor &input_tensor, const uint32_t num_q_heads, const uint32_t num_kv_heads, const uint32_t head_dim, std::vector<Tensor>& output, CoreCoord compute_with_storage_grid_size);

    struct NLPCreateHeadsDecodeDeviceOperation {
        const uint32_t num_q_heads;
        const uint32_t num_kv_heads;
        const uint32_t head_dim;
        MemoryConfig output_mem_config;

        void validate(const std::vector<Tensor>& input_tensors) const;
        std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
        std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
        operation::ProgramWithCallbacks create_program(
            const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    };
}  // namespace ttnn::operations::experimental::transformer
