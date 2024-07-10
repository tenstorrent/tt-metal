// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_eager/tt_dnn/op_library/complex/complex_ops.hpp"

namespace ttnn::operations::complex_binary_backward {

constexpr uint8_t DefaultQueueId = 0;
enum class ComplexBinaryBackwardOpType {
    COMPLEX_ADD_BW,
    COMPLEX_SUB_BW,
    COMPLEX_MUL_BW,
    COMPLEX_DIV_BW,
};

struct ComplexBinaryBackwardFunction{
static std::function<std::vector<ComplexTensor>(const ComplexTensor&, const ComplexTensor&, const ComplexTensor&, float, const MemoryConfig&)> get_function_type1(ComplexBinaryBackwardOpType OpType);
static std::function<std::vector<ComplexTensor>(const ComplexTensor&, const ComplexTensor&, const ComplexTensor&, const MemoryConfig&)> get_function_type2(ComplexBinaryBackwardOpType OpType);
};

}  // namespace ttnn::operations::complex_binary_backward
