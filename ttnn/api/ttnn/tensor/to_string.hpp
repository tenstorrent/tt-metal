// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <tt-metalium/tensor/tensor.hpp>

namespace ttnn {

std::string to_string(const tt::tt_metal::Tensor& tensor);

}  // namespace ttnn
