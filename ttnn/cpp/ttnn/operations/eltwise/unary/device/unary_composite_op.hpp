
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/ternary/where.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"

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
    LGAMMA,
    LOG1P,
    MISH,
    MULTIGAMMALN,
    SINH,
    SOFTSIGN,
    SWISH,
    TANHSHRINK,
    VAR_HW,
    STD_HW,
    NORMALIZE_HW,
    HARDSWISH,
    HARDSIGMOID,
    HARDTANH,
    SELU,
    THRESHOLD,
    GLU,
    REGLU,
    GEGLU,
    SWIGLU,
    POW,
    TRIL,
    TRIU,
    ROUND,
    POLYGAMMA,
    HARDSHRINK,
    SOFTSHRINK,
    LOGIT,
    CELU,
    LOGICAL_NOT_,
    RPOW,
    NORMALIZE_GLOBAL,
    FRAC,
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
Tensor _variance_impl(const Tensor&, const Tensor&, Tensor&, const std::optional<MemoryConfig>&);
Tensor _variance_impl(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _variance(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _std(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _std(const Tensor&, const Tensor&, Tensor&, const std::optional<MemoryConfig>&);
Tensor _std_overload(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _normalize(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _deg2rad(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _rad2deg(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _hardswish(const Tensor&, float scale =  1.0f/6.0f, float shift = 0.5f, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor _hardsigmoid(const Tensor&, float scale =  1.0f/6.0f, float shift = 0.5f, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor _hardtanh(const Tensor&, float min = -1, float max = 1, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor _selu(const Tensor&, float scale = 1.0507, float alpha = 1.67326, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor _threshold(const Tensor&, float, float, const std::optional<MemoryConfig>& );
Tensor _glu(const Tensor&, int32_t, const std::optional<MemoryConfig>& );
Tensor _reglu(const Tensor&, int32_t, const std::optional<MemoryConfig>& );
Tensor _geglu(const Tensor&, int32_t, const std::optional<MemoryConfig>& );
Tensor _swiglu(const Tensor&, int32_t, const std::optional<MemoryConfig>& );
Tensor _power(uint8_t, const Tensor&, float, const std::optional<MemoryConfig>&, std::optional<Tensor>);
Tensor _power(uint8_t, const Tensor&, uint32_t, const std::optional<MemoryConfig>&, std::optional<Tensor>);
Tensor _tril(const Tensor&, int32_t diag = 0, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor _triu(const Tensor&, int32_t diag = 0, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor _round(const Tensor&, int32_t decimal =0 , const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor _polygamma(const Tensor&, int32_t, const std::optional<MemoryConfig>& );
Tensor _hardshrink(const Tensor& a, float lambd = 0.5f, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor _softshrink(const Tensor& a, float lambd = 0.5f, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor _logit(const Tensor& a, float eps = 0.0f, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor _celu(const Tensor& a, float alpha = 1.0f, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor _logical_not_ (const Tensor&, const std::optional<MemoryConfig>&);
Tensor _rpow(const Tensor& a, float param, const std::optional<MemoryConfig>& );
Tensor _normalize_global(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _frac (const Tensor&, const std::optional<MemoryConfig>&);

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
struct OpHandler<UnaryCompositeOpType::NORMALIZE_GLOBAL> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _normalize_global(t1, mem_cfg);
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

template <>
struct OpHandler<UnaryCompositeOpType::HARDSWISH> {
    static Tensor handle(const Tensor& t1, float scale, float shift, const std::optional<MemoryConfig>& mem_cfg ) {
        return _hardswish(t1, scale, shift, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::HARDSIGMOID> {
    static Tensor handle(const Tensor& t1, float scale, float shift, const std::optional<MemoryConfig>& mem_cfg ) {
        return _hardsigmoid(t1, scale, shift, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::HARDTANH> {
    static Tensor handle(const Tensor& t1, float low, float high, const std::optional<MemoryConfig>& mem_cfg ) {
        return _hardtanh(t1, low, high, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::SELU> {
    static Tensor handle(const Tensor& t1, float scale, float alpha, const std::optional<MemoryConfig>& mem_cfg ) {
        return _selu(t1, scale, alpha, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::THRESHOLD> {
    static Tensor handle(const Tensor& t1, float threshold, float value, const std::optional<MemoryConfig>& mem_cfg ) {
        return _threshold(t1, threshold, value, mem_cfg);
    }
};

//glu (geglu, reglu, swiglu, glu) varinats are supported only for last dimension.
template <>
struct OpHandler<UnaryCompositeOpType::GLU> {
    static Tensor handle(const Tensor& t1, int32_t dim, const std::optional<MemoryConfig>& mem_cfg ) {
    return _glu(t1, dim, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::TRIL> {
    static Tensor handle(const Tensor& t1, int32_t param1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _tril(t1, param1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::TRIU> {
    static Tensor handle(const Tensor& t1, int32_t param1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _triu(t1, param1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::ROUND> {
    static Tensor handle(const Tensor& t1, int32_t param1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _round(t1, param1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::REGLU> {
    static Tensor handle(const Tensor& t1, int32_t dim, const std::optional<MemoryConfig>& mem_cfg ) {
        return _reglu(t1, dim, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::POLYGAMMA> {
    static Tensor handle(const Tensor& t1, int32_t param1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _polygamma(t1, param1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::GEGLU> {
    static Tensor handle(const Tensor& t1, int32_t dim, const std::optional<MemoryConfig>& mem_cfg ) {
        return _geglu(t1, dim, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::SWIGLU> {
    static Tensor handle(const Tensor& t1, int32_t dim, const std::optional<MemoryConfig>& mem_cfg ) {
    return _swiglu(t1, dim, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::POW> {
    static Tensor handle(uint8_t q_id, const Tensor& input, float exponent, const std::optional<MemoryConfig>& mem_cfg, std::optional<Tensor> output) {
        return _power(q_id, input, exponent, mem_cfg, output);
    }
    static Tensor handle(uint8_t q_id, const Tensor& input, uint32_t exponent, const std::optional<MemoryConfig>& mem_cfg, std::optional<Tensor> output) {
        return _power(q_id, input, exponent, mem_cfg, output);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::HARDSHRINK> {
    static Tensor handle(const Tensor& t1, float lambd, const std::optional<MemoryConfig>& mem_cfg ) {
        return _hardshrink(t1, lambd, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::SOFTSHRINK> {
    static Tensor handle(const Tensor& t1, float lambd, const std::optional<MemoryConfig>& mem_cfg ) {
        return _softshrink(t1, lambd, mem_cfg);
    }
};


template <>
struct OpHandler<UnaryCompositeOpType::LOGIT> {
    static Tensor handle(const Tensor& t1, float eps, const std::optional<MemoryConfig>& mem_cfg ) {
        return _logit(t1, eps, mem_cfg);
    }
};


template <>
struct OpHandler<UnaryCompositeOpType::CELU> {
    static Tensor handle(const Tensor& t1, float alpha, const std::optional<MemoryConfig>& mem_cfg ) {
        return _celu(t1, alpha, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::LOGICAL_NOT_> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _logical_not_(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::RPOW> {
    static Tensor handle(const Tensor& t1, float param, const std::optional<MemoryConfig>& mem_cfg ) {
        return _rpow(t1, param, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::FRAC> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg ) {
        return _frac(t1, mem_cfg);
    }
};
}
