
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <functional>
#include <optional>
#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"

namespace ttnn::operations::unary{

enum class UnaryCompositeOpType {
    DEG2RAD,
    RAD2DEG,
    ACOSH,
    ASINH,
    ATANH,
    CBRT,
    COSH,
    DIGAMMA,
    HARDSWISH,
    HARDSIGMOID,
    HARDTANH,
    LGAMMA,
    LOG1P,
    MISH,
    MULTIGAMMALN,
    SINH,
    SOFTSIGN,
    SWISH,
    TANHSHRINK,
    TRUNC,
    VAR_HW,
    STD_HW,
    NORMALIZE_HW,
};

Tensor _tanhshrink (const Tensor&, const std::optional<MemoryConfig>&);
Tensor _acosh(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _asinh(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _atanh(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _cbrt(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _cosh(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _digamma(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _lgamma(const Tensor&,  const std::optional<MemoryConfig>&);
Tensor _log1p(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _mish(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _multigammaln(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _sinh(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _softsign(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _swish(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _trunc(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _variance_impl(const Tensor&, const Tensor&, Tensor&, const std::optional<MemoryConfig>&);
Tensor _variance_impl(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _variance(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _std(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _std(const Tensor&, const Tensor&, Tensor&, const std::optional<MemoryConfig>&);
Tensor _std_overload(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _normalize(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _deg2rad(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _rad2deg(const Tensor&, const std::optional<MemoryConfig>&);

// OpHandler struct template
template <UnaryCompositeOpType OpType>
struct OpHandler;

template <>
struct OpHandler<UnaryCompositeOpType::DEG2RAD> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _deg2rad(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::RAD2DEG> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _rad2deg(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::TANHSHRINK> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _tanhshrink(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::ACOSH> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _acosh(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::ASINH> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _asinh(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::ATANH> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _atanh(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::CBRT> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _cbrt(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::COSH> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _cosh(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::DIGAMMA> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _digamma(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::LGAMMA> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _lgamma(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::LOG1P> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _log1p(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::MISH> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _mish(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::MULTIGAMMALN> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _multigammaln(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::SINH> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _sinh(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::SOFTSIGN> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _softsign(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::SWISH> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _swish(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::TRUNC> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _trunc(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::VAR_HW> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _variance(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::STD_HW> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _std_overload(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::NORMALIZE_HW> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _normalize(t1, mem_cfg);
    }
};

// Template functions to get the function pointers
template <UnaryCompositeOpType OpType>
auto get_function_type1() {
    return &OpHandler<OpType>::handle;
}
}
