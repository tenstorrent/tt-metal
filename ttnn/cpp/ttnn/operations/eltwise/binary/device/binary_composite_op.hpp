
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"

namespace ttnn::operations::binary{

enum class BinaryCompositeOpType {
    HYPOT,
    XLOGY,
    ADDALPHA,
    SUBALPHA,
    NEXTAFTER,
    ISCLOSE,
    MINIMUM,
    MAXIMUM,
    ATAN2,
    LOGICAL_XOR,
    BINARY_REMAINDER,
    BINARY_FMOD,
    DIV,
    DIV_NO_NAN,
    FLOOR_DIV,
    LOGICAL_AND_,
    LOGICAL_OR_,
    LOGICAL_XOR_,
};

Tensor _hypot(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _xlogy(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _minimum(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _maximum(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _atan2(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _logical_xor(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _nextafter(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _binary_remainder(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _binary_fmod(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _addalpha(const Tensor&, const Tensor&, float, const std::optional<MemoryConfig>&);
Tensor _subalpha(const Tensor&, const Tensor&, float, const std::optional<MemoryConfig>&);
Tensor _isclose(const Tensor&, const Tensor&, float, float, const bool, const std::optional<MemoryConfig>&);
Tensor _div(const Tensor&, const Tensor&, bool, string, const std::optional<MemoryConfig>&);
Tensor _div_overload(const Tensor&, float, bool, string, const std::optional<MemoryConfig>&);
Tensor _div_no_nan(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _div_no_nan_overload(const Tensor&, float, const std::optional<MemoryConfig>&);
Tensor _floor_div(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _floor_div_overload(const Tensor&, float, const std::optional<MemoryConfig>&);
Tensor _logical_or_(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _logical_and_(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _logical_xor_(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);

// OpHandler struct template
template <BinaryCompositeOpType OpType>
struct OpHandler;

template <BinaryCompositeOpType OpType>
struct OpHandler;

template <BinaryCompositeOpType OpType>
struct OpHandler;

template <BinaryCompositeOpType OpType>
struct OpHandler_Div;

template <BinaryCompositeOpType OpType>
struct OpHandler_Overload;


template <>
struct OpHandler<BinaryCompositeOpType::HYPOT> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, const std::optional<MemoryConfig>& mem_cfg) {
        return _hypot(t1, t2, mem_cfg);
    }
};

template <>
struct OpHandler<BinaryCompositeOpType::XLOGY> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, const std::optional<MemoryConfig>& mem_cfg) {
        return _xlogy(t1, t2, mem_cfg);
    }
};

template <>
struct OpHandler<BinaryCompositeOpType::NEXTAFTER> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, const std::optional<MemoryConfig>& mem_cfg) {
        return _nextafter(t1, t2, mem_cfg);
    }
};

template <>
struct OpHandler<BinaryCompositeOpType::MINIMUM> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, const std::optional<MemoryConfig>& mem_cfg) {
        return _minimum(t1, t2, mem_cfg);
    }
};

template <>
struct OpHandler<BinaryCompositeOpType::MAXIMUM> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, const std::optional<MemoryConfig>& mem_cfg) {
        return _maximum(t1, t2, mem_cfg);
    }
};

template <>
struct OpHandler<BinaryCompositeOpType::ATAN2> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, const std::optional<MemoryConfig>& mem_cfg) {
        return _atan2(t1, t2, mem_cfg);
    }
};

template <>
struct OpHandler<BinaryCompositeOpType::LOGICAL_XOR> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, const std::optional<MemoryConfig>& mem_cfg) {
        return _logical_xor(t1, t2, mem_cfg);
    }
};

template <>
struct OpHandler<BinaryCompositeOpType::ADDALPHA> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, float alpha, const std::optional<MemoryConfig>& mem_cfg) {
        return _addalpha(t1, t2, alpha, mem_cfg);
    }
};

template <>
struct OpHandler<BinaryCompositeOpType::SUBALPHA> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, float alpha, const std::optional<MemoryConfig>& mem_cfg) {
        return _subalpha(t1, t2, alpha, mem_cfg);
    }
};


template <>
struct OpHandler<BinaryCompositeOpType::ISCLOSE> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, float rtol, float atol, const bool equal_nan, const std::optional<MemoryConfig>& mem_cfg) {
        return _isclose(t1, t2, rtol, atol, equal_nan, mem_cfg);
    }
};


using HandleFunctionPtr1 = Tensor (*)(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
using HandleFunctionPtr2 = Tensor (*)(const Tensor&, float, const std::optional<MemoryConfig>&);
template <BinaryCompositeOpType OpType>
auto get_binary_div_like_ops() -> HandleFunctionPtr1 {
    return &OpHandler_Overload<OpType>::handle;
}
template <BinaryCompositeOpType OpType>
auto get_binary_div_like_ops_overload() -> HandleFunctionPtr2 {
    return &OpHandler_Overload<OpType>::handle;
}

using HandleFunctionPtr3 = Tensor (*)(const Tensor&, const Tensor&, bool, std::string, const std::optional<MemoryConfig>&);
using HandleFunctionPtr4 = Tensor (*)(const Tensor&, float, bool, std::string, const std::optional<MemoryConfig>&);
template <BinaryCompositeOpType OpType>
auto get_binary_div() -> HandleFunctionPtr3 {
    return &OpHandler_Div<OpType>::handle;
}
template <BinaryCompositeOpType OpType>
auto get_binary_div_overload() -> HandleFunctionPtr4 {
    return &OpHandler_Div<OpType>::handle;
}

}
