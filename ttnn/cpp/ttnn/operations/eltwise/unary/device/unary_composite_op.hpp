// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/ternary/where/where.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"

namespace ttnn::operations::unary {

enum class UnaryCompositeOpType {
    CBRT,
    DIGAMMA,
    LGAMMA,
    MULTIGAMMALN,
    SWISH,
    VAR_HW,
    STD_HW,
    NORMALIZE_HW,
    GLU,
    REGLU,
    GEGLU,
    SWIGLU,
    POW,
    TRIL,
    TRIU,
    POLYGAMMA,
    LOGIT,
    LOGICAL_NOT_,
    RPOW,
    NORMALIZE_GLOBAL,
    FRAC,
};
Tensor _cbrt(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _digamma(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _lgamma(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _multigammaln(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _swish(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _variance_impl(const Tensor&, const Tensor&, Tensor&, const std::optional<MemoryConfig>&);
Tensor _variance_impl(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _variance(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _std(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _std(const Tensor&, const Tensor&, Tensor&, const std::optional<MemoryConfig>&);
Tensor _std_overload(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _normalize(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _glu(const Tensor&, int32_t, const std::optional<MemoryConfig>&);
Tensor _reglu(const Tensor&, int32_t, const std::optional<MemoryConfig>&);
Tensor _geglu(const Tensor&, int32_t, const std::optional<MemoryConfig>&);
Tensor _swiglu(const Tensor&, int32_t, const std::optional<MemoryConfig>&);
Tensor _tril(const Tensor&, int32_t diag = 0, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor _triu(const Tensor&, int32_t diag = 0, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor _polygamma(const Tensor&, int32_t, const std::optional<MemoryConfig>&);
Tensor _logit(const Tensor& a, float eps = 0.0f, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
Tensor _logical_not_(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _rpow(const Tensor& a, float param, const std::optional<MemoryConfig>&);
Tensor _normalize_global(const Tensor&, const std::optional<MemoryConfig>&);
Tensor _frac(const Tensor&, const std::optional<MemoryConfig>&);

// OpHandler struct template
template <UnaryCompositeOpType OpType>
struct OpHandler;

template <>
struct OpHandler<UnaryCompositeOpType::CBRT> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg) { return _cbrt(t1, mem_cfg); }
};

template <>
struct OpHandler<UnaryCompositeOpType::NORMALIZE_GLOBAL> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg) {
        return _normalize_global(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::DIGAMMA> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg) { return _digamma(t1, mem_cfg); }
};

template <>
struct OpHandler<UnaryCompositeOpType::LGAMMA> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg) { return _lgamma(t1, mem_cfg); }
};

template <>
struct OpHandler<UnaryCompositeOpType::MULTIGAMMALN> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg) {
        return _multigammaln(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::SWISH> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg) { return _swish(t1, mem_cfg); }
};

template <>
struct OpHandler<UnaryCompositeOpType::VAR_HW> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg) {
        return _variance(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::STD_HW> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg) {
        return _std_overload(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::NORMALIZE_HW> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg) {
        return _normalize(t1, mem_cfg);
    }
};

// glu (geglu, reglu, swiglu, glu) varinats are supported only for last dimension.
template <>
struct OpHandler<UnaryCompositeOpType::GLU> {
    static Tensor handle(const Tensor& t1, int32_t dim, const std::optional<MemoryConfig>& mem_cfg) {
        return _glu(t1, dim, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::TRIL> {
    static Tensor handle(const Tensor& t1, int32_t param1, const std::optional<MemoryConfig>& mem_cfg) {
        return _tril(t1, param1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::TRIU> {
    static Tensor handle(const Tensor& t1, int32_t param1, const std::optional<MemoryConfig>& mem_cfg) {
        return _triu(t1, param1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::REGLU> {
    static Tensor handle(const Tensor& t1, int32_t dim, const std::optional<MemoryConfig>& mem_cfg) {
        return _reglu(t1, dim, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::POLYGAMMA> {
    static Tensor handle(const Tensor& t1, int32_t param1, const std::optional<MemoryConfig>& mem_cfg) {
        return _polygamma(t1, param1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::GEGLU> {
    static Tensor handle(const Tensor& t1, int32_t dim, const std::optional<MemoryConfig>& mem_cfg) {
        return _geglu(t1, dim, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::SWIGLU> {
    static Tensor handle(const Tensor& t1, int32_t dim, const std::optional<MemoryConfig>& mem_cfg) {
        return _swiglu(t1, dim, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::LOGIT> {
    static Tensor handle(const Tensor& t1, float eps, const std::optional<MemoryConfig>& mem_cfg) {
        return _logit(t1, eps, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::LOGICAL_NOT_> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg) {
        return _logical_not_(t1, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::RPOW> {
    static Tensor handle(const Tensor& t1, float param, const std::optional<MemoryConfig>& mem_cfg) {
        return _rpow(t1, param, mem_cfg);
    }
};

template <>
struct OpHandler<UnaryCompositeOpType::FRAC> {
    static Tensor handle(const Tensor& t1, const std::optional<MemoryConfig>& mem_cfg) { return _frac(t1, mem_cfg); }
};
}  // namespace ttnn::operations::unary
