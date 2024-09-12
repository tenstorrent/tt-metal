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
    SUBALPHA_BW,
    SUB_BW,
    XLOGY_BW,
    HYPOT_BW,
    LDEXP_BW,
    LOGADDEXP_BW,
    LOGADDEXP2_BW,
    SQUARED_DIFFERENCE_BW,
    ADD_BW,
    ASSIGN_BW,
    CONCAT_BW,
    RSUB_BW,
    BIAS_GELU_BW,
    MIN_BW,
    MAX_BW,
    DIV_BW,
    MUL_BW,
    REMAINDER_BW,
    FMOD_BW,
};

std::vector<Tensor> _atan2_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _rsub_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _xlogy_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _hypot_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _ldexp_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _logaddexp_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _sub_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _gt_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _logaddexp2_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
std::vector<Tensor> _squared_difference_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);
template <bool>
std::vector<Tensor> _min_or_max_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config);

std::vector<ttnn::Tensor> _subalpha_bw( const Tensor& grad, const Tensor& input, const Tensor& other, float alpha = 1.0f, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<ttnn::Tensor> _div_bw( const Tensor& grad, const Tensor& input, const Tensor& other, string round_mode = "None" , const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<ttnn::Tensor> _concat_bw( const Tensor& grad, const Tensor& input, const Tensor& other, int dim = 0, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<std::optional<ttnn::Tensor>> _eq_bw(uint8_t queue_id, const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config, const std::vector<bool>& are_required_outputs, std::optional<Tensor> input_grad, std::optional<Tensor> other_grad);

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
struct OpHandler<BinaryBackwardOpType::RSUB_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _rsub_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler<BinaryBackwardOpType::SUB_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, const std::optional<MemoryConfig>& output_mem_config ) {
        return _sub_bw(grad, input, other, output_mem_config);
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

template <>
struct OpHandler<BinaryBackwardOpType::SUBALPHA_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const std::optional<MemoryConfig>& output_mem_config ) {
        return _subalpha_bw(grad, input, other, alpha, output_mem_config);
    }
};

template <>
struct OpHandler<BinaryBackwardOpType::DIV_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, string round_mode, const std::optional<MemoryConfig>& output_mem_config ) {
        return _div_bw(grad, input, other, round_mode, output_mem_config);
    }
};

template <>
struct OpHandler<BinaryBackwardOpType::CONCAT_BW> {
    static std::vector<Tensor> handle( const Tensor& grad, const Tensor& input, const Tensor& other, int dim, const std::optional<MemoryConfig>& output_mem_config ) {
        return _concat_bw(grad, input, other, dim, output_mem_config);
    }
};


}  // namespace ttnn::operations::binary_backward
