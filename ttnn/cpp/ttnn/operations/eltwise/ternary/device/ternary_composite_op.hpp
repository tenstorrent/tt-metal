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

};

Tensor _addcmul(const Tensor&, const Tensor&, const Tensor&, float, const std::optional<MemoryConfig>&);
Tensor _addcdiv(const Tensor&, const Tensor&, const Tensor&, float, const std::optional<MemoryConfig>&);


template <TernaryCompositeOpType OpType>
struct OpHandler_Float;


template <>
struct OpHandler_Float<TernaryCompositeOpType::ADDCMUL> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, const Tensor& t3, float value, const std::optional<MemoryConfig>& mem_cfg) {
        return _addcmul(t1, t2, t3, alpha, mem_cfg);
    }
};

template <>
struct OpHandler_Float<TernaryCompositeOpType::ADDCDIV> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, const Tensor& t3, float value, const std::optional<MemoryConfig>& mem_cfg) {
        return _addcdiv(t1, t2, t3, alpha, mem_cfg);
    }
};

template <TernaryCompositeOpType OpType>
auto get_function_type0() {
    return &OpHandler_Float<OpType>::handle;
}


}
