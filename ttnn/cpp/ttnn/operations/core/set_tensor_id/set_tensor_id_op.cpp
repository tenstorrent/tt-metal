// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "set_tensor_id_op.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::core {

std::uint64_t SetTensorId::invoke(std::uint64_t id) {
    tt::tt_metal::Tensor::tensor_id_counter.store(id);
    return id;
}

}  // namespace ttnn::operations::core
