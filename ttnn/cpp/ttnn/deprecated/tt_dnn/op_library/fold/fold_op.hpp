// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

namespace tt::tt_metal {

enum class FoldOpParallelizationStrategy { SINGLE_CORE, SHARDED_MULTI_CORE };

struct Fold {
    uint8_t stride_h;
    uint8_t stride_w;
    bool is_sharded;

    void validate(const std::vector<Tensor> &input_tensors) const;

    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;

    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    FoldOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
};

operation::ProgramWithCallbacks fold_single_core(
    const Tensor &input, const Tensor &output, uint8_t stride_h, uint8_t stride_w);

operation::ProgramWithCallbacks fold_multi_core(
    const Tensor &input, const Tensor &output, uint8_t stride_h, uint8_t stride_w);

Tensor fold(const Tensor &input_tensor_a, uint8_t stride_h, uint8_t stride_w, bool use_transpose_as_fold = false, const std::optional<const Shape>& output_shape = std::nullopt, uint8_t pad_c = 0, uint8_t pad_h = 0, uint8_t pad_w = 0);
}  // namespace tt::tt_metal
