// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fetch_and_increment_tensor_id_op.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::core {

std::uint64_t FetchAndIncrementTensorId::invoke() { return tt::tt_metal::Tensor::tensor_id_counter.fetch_add(1); }

}  // namespace ttnn::operations::core
