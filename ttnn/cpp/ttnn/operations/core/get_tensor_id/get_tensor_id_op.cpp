// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "get_tensor_id_op.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::core {

std::uint64_t GetTensorId::invoke() { return tt::tt_metal::Tensor::tensor_id_counter.load(); }

}  // namespace ttnn::operations::core
