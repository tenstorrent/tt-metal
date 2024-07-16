// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <functional>
#include <optional>
#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"

namespace ttnn::operations::ternary{

enum class TernaryCompositeOpType {
    ADDCMUL,
    ADDCDIV,
    LERP,

};

Tensor _addcmul(const Tensor&, const Tensor&, const Tensor&, float, const std::optional<MemoryConfig>&);
Tensor _addcdiv(const Tensor&, const Tensor&, const Tensor&, float, const std::optional<MemoryConfig>&);
Tensor _lerp(const Tensor&, const Tensor&, const Tensor&, float, const std::optional<MemoryConfig>&);
Tensor _lerp_overload(const Tensor&, const Tensor&, const Tensor&, float, const std::optional<MemoryConfig>&);


template <TernaryCompositeOpType OpType>
struct OpHandler_Float;

template <TernaryCompositeOpType OpType>
struct OpHandler_Lerp;


template <>
struct OpHandler_Float<TernaryCompositeOpType::ADDCMUL> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, const Tensor& t3, const std::optional<MemoryConfig>& mem_cfg) {
        return _addcmul(t1, t2, t3, mem_cfg);
    }
};

template <>
struct OpHandler_Float<TernaryCompositeOpType::ADDCDIV> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, const Tensor& t3, float value, const std::optional<MemoryConfig>& mem_cfg) {
        return _addcdiv(t1, t2, t3, alpha, mem_cfg);
    }
};

template <>
struct OpHandler_Lerp<TernaryCompositeOpType::LERP> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, const Tensor& t3, const std::optional<MemoryConfig>& mem_cfg) {
        return _lerp(t1, t2, t3, mem_cfg);
    }
    static Tensor handle(const Tensor& t1, const Tensor& t2, float value, const std::optional<MemoryConfig>& mem_cfg) {
        return _lerp_overload(t1, t2, value, mem_cfg);
    }
};

template <TernaryCompositeOpType OpType>
auto get_function_type_ternary_with_float() {
    return &OpHandler_Float<OpType>::handle;
}

template <TernaryCompositeOpType OpType>
auto get_function_type_lerp() {
    return &OpHandler_Lerp<OpType>::handle;
}


} // namespace ttnn::operations::ternary
