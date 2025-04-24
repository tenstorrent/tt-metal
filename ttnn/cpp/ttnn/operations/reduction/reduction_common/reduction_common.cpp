// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduction_common.hpp"

#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/core/core.hpp"

namespace reduction_common {

ttnn::Tensor perform_transpose(
    const ttnn::Tensor& input_tensor, const bool is_dim_last_idx, const int8_t dim1, const int8_t dim2) {
    return is_dim_last_idx ? input_tensor : ttnn::transpose(input_tensor, dim1, dim2, input_tensor.memory_config());
}

ttnn::Tensor transform_to_4d_tensor(const ttnn::Tensor& input_tensor, const bool is_rank_le_4d) {
    return is_rank_le_4d ? ttnn::operations::core::unsqueeze_to_4D(input_tensor)
                         : ttnn::operations::data_movement::squeeze_from_ND_to_4D(input_tensor);
}

}  // namespace reduction_common
