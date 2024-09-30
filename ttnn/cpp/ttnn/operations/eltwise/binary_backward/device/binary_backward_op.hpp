// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::binary_backward {

enum class BinaryBackwardOpType {
    ATAN2_BW,
    ADDALPHA_BW,
    XLOGY_BW,
    HYPOT_BW,
    LDEXP_BW,
    LOGADDEXP_BW,
    LOGADDEXP2_BW,
    SQUARED_DIFFERENCE_BW,
    ASSIGN_BW,
    BIAS_GELU_BW,
    MIN_BW,
    MAX_BW,
    REMAINDER_BW,
    FMOD_BW,
};

std::vector<Tensor> _atan2_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _xlogy_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _hypot_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _ldexp_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _logaddexp_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _logaddexp2_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _squared_difference_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
template <bool>
std::vector<Tensor> _min_or_max_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);

// OpHandler struct template
template <BinaryBackwardOpType OpType>
struct OpHandler;


template <>
struct OpHandler<BinaryBackwardOpType::ATAN2_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _atan2_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler<BinaryBackwardOpType::XLOGY_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _xlogy_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler<BinaryBackwardOpType::HYPOT_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _hypot_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler<BinaryBackwardOpType::LDEXP_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _ldexp_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler<BinaryBackwardOpType::LOGADDEXP_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _logaddexp_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler<BinaryBackwardOpType::LOGADDEXP2_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _logaddexp2_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler<BinaryBackwardOpType::SQUARED_DIFFERENCE_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _squared_difference_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler<BinaryBackwardOpType::MIN_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _min_or_max_bw<false>(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler<BinaryBackwardOpType::MAX_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _min_or_max_bw<true>(grad, input, other, output_mem_config);
    }
};


}  // namespace ttnn::operations::binary_backward
