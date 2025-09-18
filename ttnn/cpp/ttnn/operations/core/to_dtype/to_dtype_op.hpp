// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::core {

struct ToDtype {
    static Tensor invoke(const ttnn::Tensor& input_tensor, const ttnn::DataType& dtype);
};

}  // namespace ttnn::operations::core
