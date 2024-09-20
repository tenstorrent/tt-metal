
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/operations/core/core.hpp"

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
    DIV_NO_NAN,
    FLOOR_DIV,
    LOGICAL_XOR_,
    SCATTER,
    OUTER,
    POLYVAL,
};

Tensor _hypot(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _xlogy(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _minimum(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _maximum(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _atan2(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _logical_xor(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _nextafter(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _addalpha(const Tensor&, const Tensor&, float, const std::optional<MemoryConfig>&);
Tensor _subalpha(const Tensor&, const Tensor&, float, const std::optional<MemoryConfig>&);
Tensor _isclose(const Tensor&, const Tensor&, float, float, const bool, const std::optional<MemoryConfig>&);
Tensor _div_no_nan(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _div_no_nan_overload(const Tensor&, float, const std::optional<MemoryConfig>&);
Tensor _floor_div(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _floor_div_overload(const Tensor&, float, const std::optional<MemoryConfig>&);
Tensor _logical_xor_(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _scatter(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _outer(const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _polyval(const Tensor&, const std::vector<float>&, const std::optional<MemoryConfig>&);


// OpHandler struct template
template <BinaryCompositeOpType OpType>
struct OpHandler;

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
struct OpHandler<BinaryCompositeOpType::LOGICAL_XOR_> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, const std::optional<MemoryConfig>& mem_cfg) {
        return _logical_xor_(t1, t2, mem_cfg);
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

template <>
struct OpHandler<BinaryCompositeOpType::DIV_NO_NAN> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, const std::optional<MemoryConfig>& mem_cfg) {
        return _div_no_nan(t1, t2, mem_cfg);
    }
    static Tensor handle(const Tensor& t1, float value, const std::optional<MemoryConfig>& mem_cfg) {
        return _div_no_nan_overload(t1, value, mem_cfg);
    }
};

template <>
struct OpHandler<BinaryCompositeOpType::FLOOR_DIV> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, const std::optional<MemoryConfig>& mem_cfg) {
        return _floor_div(t1, t2, mem_cfg);
    }
    static Tensor handle(const Tensor& t1, float value, const std::optional<MemoryConfig>& mem_cfg) {
        return _floor_div_overload(t1, value, mem_cfg);
    }
};

template <>
struct OpHandler<BinaryCompositeOpType::SCATTER> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, const std::optional<MemoryConfig>& mem_cfg) {
        return _scatter(t1, t2, mem_cfg);
    }
};

template <>
struct OpHandler<BinaryCompositeOpType::OUTER> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, const std::optional<MemoryConfig>& mem_cfg) {
        return _outer(t1, t2, mem_cfg);
    }
};

template <>
struct OpHandler<BinaryCompositeOpType::POLYVAL> {
    static Tensor handle(const Tensor& t1, const std::vector<float>& coeffs, const std::optional<MemoryConfig>& mem_cfg) {
        return _polyval(t1, coeffs, mem_cfg);
    }
};

}
