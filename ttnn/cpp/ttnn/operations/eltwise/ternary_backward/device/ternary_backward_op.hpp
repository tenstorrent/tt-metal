// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

namespace ttnn::operations::ternary_backward {

using OptionalTensor = std::optional<Tensor>;

enum class TernaryBackwardOpType {
    ADDCMUL_BW,
    ADDCDIV_BW,
    WHERE_BW,
    LERP_BW,
};

std::vector<Tensor> _lerp_overload(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&);
std::vector<Tensor> _addcmul_bw(const Tensor&, const Tensor&, const Tensor&, const Tensor&, float, const MemoryConfig&);
std::vector<Tensor> _addcdiv_bw(const Tensor&, const Tensor&, const Tensor&, const Tensor&, float, const MemoryConfig&);
std::vector<OptionalTensor> _where_bw(uint8_t, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&, const std::vector<bool>&, OptionalTensor, OptionalTensor);

// OpHandler struct template
template <TernaryBackwardOpType OpType>
struct OpHandler;

template <>
struct OpHandler<TernaryBackwardOpType::LERP_BW> {
    static std::vector<Tensor> handle(const Tensor& t1, const Tensor& t2, const Tensor& t3, const Tensor& t4,
                                        const MemoryConfig& mem_cfg) {
        return _lerp_overload(t1, t2, t3, t4, mem_cfg);
    }
};

template <>
struct OpHandler<TernaryBackwardOpType::ADDCMUL_BW> {
    static std::vector<Tensor> handle(const Tensor& t1, const Tensor& t2, const Tensor& t3, const Tensor& t4,
                                        float value, const MemoryConfig& mem_cfg) {
        return _addcmul_bw(t1, t2, t3, t4, value, mem_cfg);
    }
};

template <>
struct OpHandler<TernaryBackwardOpType::ADDCDIV_BW> {
    static std::vector<Tensor> handle(const Tensor& t1, const Tensor& t2, const Tensor& t3, const Tensor& t4,
                                        float value, const MemoryConfig& mem_cfg) {
        return _addcdiv_bw(t1, t2, t3, t4, value, mem_cfg);
    }
};

template <>
struct OpHandler<TernaryBackwardOpType::WHERE_BW> {
    static std::vector<OptionalTensor> handle(uint8_t queue_id, const Tensor& t1, const Tensor& t2, const Tensor& t3,
                                                     const Tensor& t4, const MemoryConfig& mem_cfg, const std::vector<bool>& are_required_outputs,
                                                     OptionalTensor input_grad, OptionalTensor other_grad) {
        return _where_bw(queue_id, t1, t2, t3, t4, mem_cfg, are_required_outputs, input_grad, other_grad);
    }
};

}  // namespace ttnn::operations::ternary_backward
