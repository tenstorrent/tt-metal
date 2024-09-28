// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"

namespace ttnn::operations::unary_backward {

enum class UnaryBackwardOpType {
    ADD_BW,
    EQ_BW,
    GT_BW,
    SUB_BW,
};

std::vector<Tensor> _sub_bw( const Tensor& grad, const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _gt_bw( const Tensor& grad, const Tensor& input, float other, const std::optional<MemoryConfig>& output_mem_config);

std::vector<Tensor> _add_bw( const Tensor& grad, const Tensor& input, float alpha, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
std::vector<Tensor> _eq_bw( const Tensor& grad, const Tensor& input, float other, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

Tensor change_layout_to_tile(const Tensor& temp, const MemoryConfig& output_mem_config);

// OpHandler struct template
template <UnaryBackwardOpType OpType>
struct OpHandler;

template <>
struct OpHandler<UnaryBackwardOpType::GT_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _gt_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::SUB_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config ) {
        return _sub_bw(grad, input, scalar, output_mem_config);
    }
};

template <>
struct OpHandler<UnaryBackwardOpType::EQ_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, float other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _eq_bw(grad, input, other, output_mem_config);
    }
};

}  // namespace ttnn::operations::unary_backward
