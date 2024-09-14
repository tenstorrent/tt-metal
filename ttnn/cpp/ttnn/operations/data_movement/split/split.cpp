// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "device/split_op.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape/reshape.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/split/split.hpp"


namespace ttnn::operations::data_movement {


namespace detail {

    std::vector<Tensor> impl_split_last_dim_n_chunks_tiled(const Tensor &input_tensor, const MemoryConfig &mem_config, const int64_t &num_chunks) {
        auto input_shape = input_tensor.get_legacy_shape();
        auto padded_input_shape = ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(input_shape);
        ttnn::operations::experimental::auto_format::FormatParams input_format_params = {.pad_shape = padded_input_shape, .pad_value = 0.0, .target_layout = Layout::TILE};
        return operation::run_with_autoformat(SplitDeviceOperation{num_chunks, 3, mem_config}, {input_tensor}, {input_format_params}, {Layout::TILE, Layout::TILE});
    }

    std::vector<Tensor> split_last_dim_n_chunks_tiled(const Tensor &input_tensor, const MemoryConfig &mem_config, const int64_t &num_chunks) {
        const auto shape = input_tensor.get_legacy_shape();
        const bool pre_post_reshape = shape[0] > 1;

        if (!pre_post_reshape) {
            return impl_split_last_dim_n_chunks_tiled(input_tensor, mem_config, num_chunks);
        }

        const int W = 1, Z = shape[0] * shape[1], Y = shape[2], X = shape[3];
        const Tensor &reshaped_tensor = ttnn::reshape_on_device(input_tensor, 1, -1, Y, X, mem_config);

        auto part_reshaped = impl_split_last_dim_n_chunks_tiled(reshaped_tensor, mem_config, num_chunks);

        std::vector<Tensor> results;
        results.reserve(part_reshaped.size());
        for (auto &part : part_reshaped) results.emplace_back(ttnn::reshape_on_device(part, -1, shape[1], Y, X / 2, mem_config));

        return results;
    }


std::vector<Tensor> split_dim_n_chunks_tiled(
    const Tensor &input_tensor, int dim /* = 3 */, const MemoryConfig &mem_config /* = default */, const int64_t &num_chunks /* = 2 */) {
    if (dim == 3) {
        return split_last_dim_n_chunks_tiled(input_tensor, mem_config, num_chunks);
    }
    Tensor ref_input_tensor = ttnn::transpose(input_tensor, dim, 3, mem_config);
    auto transposed_result = split_last_dim_n_chunks_tiled(ref_input_tensor, mem_config, num_chunks);
    std::vector<Tensor> results;
    results.reserve(transposed_result.size());
    for (Tensor &t : transposed_result) {
        results.emplace_back(ttnn::transpose(t, dim, 3, mem_config));
    }
    return results;
}

}


std::vector<ttnn::Tensor> SplitOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    int64_t& num_splits,
    int64_t& dim,
    const std::optional<MemoryConfig>& memory_config_arg) {
        auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
        return detail::split_dim_n_chunks_tiled(input_tensor, dim, memory_config, num_splits);
}

std::vector<ttnn::Tensor> SplitOperation::invoke(
    const ttnn::Tensor& input_tensor,
    int64_t& num_splits,
    int64_t& dim,
    const std::optional<MemoryConfig>& memory_config) {
        return invoke(DefaultQueueId, input_tensor, num_splits, dim, memory_config);
}

std::vector<ttnn::Tensor> SplitOperation::invoke(const ttnn::Tensor& input_tensor, int64_t& num_splits, int64_t& dim) {
    return invoke(DefaultQueueId, input_tensor, num_splits, dim, std::nullopt);
}

} // ttnn::operations::data_movement namespace
