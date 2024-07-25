// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/types.hpp"
#include "tt_metal/common/bfloat16.hpp"

#include "where_op.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

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
        return _addcmul(t1, t2, t3, value, mem_cfg);
    }
};

template <>
struct OpHandler_Float<TernaryCompositeOpType::ADDCDIV> {
    static Tensor handle(const Tensor& t1, const Tensor& t2, const Tensor& t3, float value, const std::optional<MemoryConfig>& mem_cfg) {
        return _addcdiv(t1, t2, t3, value, mem_cfg);
    }
};

template <TernaryCompositeOpType OpType>
auto get_ternary_fn_float() {
    return &OpHandler_Float<OpType>::handle;
}

}
