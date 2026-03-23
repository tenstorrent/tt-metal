// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

ttnn::SmallVector<int> generate_reduce_dim(
    const ttnn::Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttnn::SmallVector<int>>>& dim_arg) {
    const auto& input_shape = input_tensor_arg.logical_shape();
    auto rank = input_shape.rank();
    ttnn::SmallVector<int> dim{};
    if (dim_arg.has_value()) {
        if (std::holds_alternative<int>(dim_arg.value())) {
            auto dim_as_int = std::get<int>(dim_arg.value());
            dim = ttnn::SmallVector<int>({dim_as_int});
        } else if (std::holds_alternative<int64_t>(dim_arg.value())) {
            auto dim_as_int64 = std::get<int64_t>(dim_arg.value());
            TT_FATAL(
                dim_as_int64 >= std::numeric_limits<int>::lowest() && dim_as_int64 <= std::numeric_limits<int>::max(),
                "Dimension must be in the range [{}, {}]",
                std::numeric_limits<int>::lowest(),
                std::numeric_limits<int>::max());
            dim = ttnn::SmallVector<int>({static_cast<int>(dim_as_int64)});
        } else {
            dim = std::get<ttnn::SmallVector<int>>(dim_arg.value());
        }
    }
    if (dim.empty()) {
        dim = ttnn::SmallVector<int>(rank);
        for (int i = 0; i < rank; i++) {
            dim[i] = i;
        }
        // It's already sorted and all are non-negative.
        return dim;
    }

    for (int i = 0; i < dim.size(); i++) {
        int& dim_i = dim[i];
        if (dim_i < 0) {
            dim_i += rank;
            if (rank == 0 && dim_i == -1) {
                // Special case for rank 0 tensor (scalar) with dim=-1.
                // While scalar technically has no dimensions, set dim to 0 because
                // that's the cleanest way to make ttnn compatible with PyTorch.
                dim_i = 0;
            }
        }
        TT_FATAL(
            (dim_i >= 0 && dim_i < rank) || (rank == 0 && dim_i == 0),
            "Unsupported dim {} at index {}. After possible adjustment, needs to be at least 0 and less than rank {}, "
            "or exactly 0 for a rank 0 tensor.",
            dim_i,
            i,
            rank);
    }

    std::sort(dim.begin(), dim.end());
    return dim;
}

}  // namespace reduction_common
