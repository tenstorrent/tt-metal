// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::binary_backward {

constexpr uint8_t DefaultQueueId = 0;
enum class BinaryBackwardOpType {
    ATAN2_BW,
    EMBEDDING_BW,
    ADDALPHA_BW,
    SUBALPHA_BW,
    SUB_BW,
    XLOGY_BW,
    HYPOT_BW,
    LDEXP_BW,
    LOGADDEXP_BW,
    LOGADDEXP2_BW,
    SQUARED_DIFFERENCE_BW,
    ADD_BW,
    BINARY_EQ_BW,
    BINARY_ASSIGN_BW,
    CONCAT_BW,
    BINARY_LE_BW,
    RSUB_BW,
    BIAS_GELU_BW,
    BINARY_GT_BW,
    BINARY_LT_BW,
    BINARY_NE_BW,
    BINARY_GE_BW,
    MIN_BW,
    MAX_BW,
    DIV_BW,
    LERP_BW,
    MUL_BW,
};

namespace utils {
    std::function<std::vector<ttnn::Tensor>(const ttnn::Tensor&, const ttnn::Tensor&, const ttnn::Tensor&, const ttnn::MemoryConfig&)> get_function_type1(BinaryBackwardOpType OpType);
    std::function<std::vector<ttnn::Tensor>(const ttnn::Tensor&, const ttnn::Tensor&, const ttnn::Tensor&, float, const ttnn::MemoryConfig&)> get_function_type1_w_float(BinaryBackwardOpType OpType);
    std::function<std::vector<ttnn::Tensor>(const ttnn::Tensor&, const ttnn::Tensor&, const ttnn::Tensor&, std::string, const ttnn::MemoryConfig&)> get_function_type1_w_string(BinaryBackwardOpType OpType);
    std::function<std::vector<std::optional<ttnn::Tensor>>(uint8_t , const ttnn::Tensor&, const ttnn::Tensor&, const Tensor&, float, const ttnn::MemoryConfig&, const std::vector<bool>&, std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>)> get_function_type2(BinaryBackwardOpType OpType);
    std::function<std::vector<std::optional<ttnn::Tensor>>(const ttnn::Tensor&, const Tensor&, const Tensor&, float, const ttnn::MemoryConfig&, const std::vector<bool>&, std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>)> get_function_type2_wo_qid(BinaryBackwardOpType OpType);
    std::function<std::vector<std::optional<ttnn::Tensor>>(uint8_t , const ttnn::Tensor&, const Tensor&, const Tensor&, const ttnn::MemoryConfig&, const std::vector<bool>&, std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>)> get_function_type3(BinaryBackwardOpType OpType);
    std::function<std::vector<std::optional<ttnn::Tensor>>(const ttnn::Tensor&, const Tensor&, const ttnn::Tensor&, const ttnn::MemoryConfig&, const std::vector<bool>&, std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>)> get_function_type3_wo_qid(BinaryBackwardOpType OpType);
    std::function<std::vector<ttnn::Tensor>(const ttnn::Tensor&, const ttnn::Tensor&, const ttnn::Tensor&, const ttnn::Tensor&, const ttnn::MemoryConfig&)> get_overload_function(BinaryBackwardOpType OpType);
}


}  // namespace ttnn::operations::binary
