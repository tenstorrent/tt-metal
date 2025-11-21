// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::detail {

uint32_t get_num_cores_channels_from_sharded_tensor(const Tensor& tensor);

}  // namespace ttnn::operations::experimental::detail
