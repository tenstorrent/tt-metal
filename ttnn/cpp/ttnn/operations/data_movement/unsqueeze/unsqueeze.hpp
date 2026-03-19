// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"

namespace ttnn {

ttnn::Tensor unsqueeze(const ttnn::Tensor& input_tensor, int dim);

}  // namespace ttnn
