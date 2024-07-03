
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
    TRIL,
    TRIU,
};
struct UnaryCompositeFunction
{
    static std::function<Tensor(const Tensor&, const std::optional<MemoryConfig>&)> get_function_type1(UnaryCompositeOpType OpType);
    static std::function<Tensor(const Tensor&, float, float, const std::optional<MemoryConfig>&)> get_function_type2(UnaryCompositeOpType OpType);
    static std::function<Tensor(const Tensor&, int, const std::optional<MemoryConfig>&)> get_function_type3(UnaryCompositeOpType OpType);
};
}
