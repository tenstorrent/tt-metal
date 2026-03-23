// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

namespace ttnn::operations::reduction::detail {

template <PoolOpType reduce_op, ReduceOpDim reduce_dim>
struct ReduceOpWithPadding {
    MemoryConfig output_mem_config;
    const DataType output_dtype;
    std::optional<const ttnn::Shape> output_shape;
    float pad_value;
    ttnn::Shape original_shape;

    void validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<ttnn::Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple("output_mem_config", "output_dtype", "output_shape", "pad_value", "original_shape");
    const auto attribute_values() const {
        return std::forward_as_tuple(std::cref(this->output_mem_config), std::cref(this->output_dtype), std::cref(this->output_shape), std::cref(this->pad_value), std::cref(this->original_shape));
    }

    const operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

using ReduceHWithPadding = ReduceOpWithPadding<PoolOpType::SUM, ReduceOpDim::H>;
using ReduceWWithPadding = ReduceOpWithPadding<PoolOpType::SUM, ReduceOpDim::W>;
using ReduceHWWithPadding = ReduceOpWithPadding<PoolOpType::SUM, ReduceOpDim::HW>;

}  // namespace ttnn::operations::reduction::detail