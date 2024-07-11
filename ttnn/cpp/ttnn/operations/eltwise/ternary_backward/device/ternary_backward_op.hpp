// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "tensor/tensor.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

namespace ttnn::operations::ternary_backward {

enum class TernaryBackwardOpType {
    ADDCMUL_BW,
    ADDCDIV_BW,
    WHERE_BW,
    LERP_BW,
};

struct TernaryBackwardFunction{
    static std::function<std::vector<Tensor>(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&)> get_function_type0(TernaryBackwardOpType OpType);
    static std::function<std::vector<Tensor>(const Tensor&, const Tensor&, const Tensor&, const Tensor&, float, const MemoryConfig&)> get_function_type(TernaryBackwardOpType OpType);
    static std::function<std::vector<std::optional<Tensor>>(uint8_t , const Tensor&, const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&, const std::vector<bool>&, std::optional<Tensor>, std::optional<Tensor>)> get_function_type_opt(TernaryBackwardOpType OpType);
    static std::function<std::vector<std::optional<Tensor>>(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&, const std::vector<bool>&, std::optional<Tensor>, std::optional<Tensor>)> get_function_type_opt_wo_qid(TernaryBackwardOpType OpType);
};

}  // namespace ttnn::operations::ternary_backward
