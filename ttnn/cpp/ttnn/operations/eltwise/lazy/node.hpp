// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/lazy/operation.hpp"
#include "ttnn/operations/eltwise/lazy/param.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <variant>

namespace ttnn::operations::lazy {

struct FunctionNode {
    Operation operation;
    Arguments<std::size_t> offsets;
    Params params;
};

using Node = std::variant<Tensor, FunctionNode>;

}  // namespace ttnn::operations::lazy
