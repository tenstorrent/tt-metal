// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <magic_enum/magic_enum.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <functional>
#include <optional>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "where.hpp"

namespace tt {
namespace tt_metal {
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn::operations::ternary {

enum class TernaryCompositeOpType {
    ADDCMUL,
    ADDCDIV,
    LERP,
    MAC,
};

Tensor _addcmul(const Tensor&, const Tensor&, const Tensor&, float, const std::optional<MemoryConfig>&);
Tensor _addcdiv(const Tensor&, const Tensor&, const Tensor&, float, const std::optional<MemoryConfig>&);
Tensor _lerp(const Tensor&, const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _lerp_overload(const Tensor&, const Tensor&, float, const std::optional<MemoryConfig>&);
Tensor _mac(const Tensor&, const Tensor&, const Tensor&, const std::optional<MemoryConfig>&);
Tensor _mac_overload(const Tensor&, float, float, const std::optional<MemoryConfig>&);

template <TernaryCompositeOpType OpType>
struct OpHandler;

template <>
struct OpHandler<TernaryCompositeOpType::ADDCMUL> {
    static Tensor handle(
        const Tensor& t1, const Tensor& t2, const Tensor& t3, float value, const std::optional<MemoryConfig>& mem_cfg) {
        return _addcmul(t1, t2, t3, value, mem_cfg);
    }
};

template <>
struct OpHandler<TernaryCompositeOpType::ADDCDIV> {
    static Tensor handle(
        const Tensor& t1, const Tensor& t2, const Tensor& t3, float value, const std::optional<MemoryConfig>& mem_cfg) {
        return _addcdiv(t1, t2, t3, value, mem_cfg);
    }
};

template <>
struct OpHandler<TernaryCompositeOpType::LERP> {
    static Tensor handle(
        const Tensor& t1, const Tensor& t2, const Tensor& t3, const std::optional<MemoryConfig>& mem_cfg) {
        return _lerp(t1, t2, t3, mem_cfg);
    }
    static Tensor handle(const Tensor& t1, const Tensor& t2, float value, const std::optional<MemoryConfig>& mem_cfg) {
        return _lerp_overload(t1, t2, value, mem_cfg);
    }
};

template <>
struct OpHandler<TernaryCompositeOpType::MAC> {
    static Tensor handle(
        const Tensor& t1, const Tensor& t2, const Tensor& t3, const std::optional<MemoryConfig>& mem_cfg) {
        return _mac(t1, t2, t3, mem_cfg);
    }
    static Tensor handle(const Tensor& t1, float value1, float value2, const std::optional<MemoryConfig>& mem_cfg) {
        return _mac_overload(t1, value1, value2, mem_cfg);
    }
};

}  // namespace ttnn::operations::ternary
