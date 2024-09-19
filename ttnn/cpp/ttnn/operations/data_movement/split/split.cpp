// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/run_operation.hpp"
#include "device/split_op.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape/reshape.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/data_movement/split/split.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"


namespace ttnn::operations::data_movement {


namespace detail {

    std::vector<Tensor> split_dim_n_chunks_rm(const Tensor &input_tensor, int dim, int num_splits, const MemoryConfig &mem_config) {
        TT_FATAL(input_tensor.get_layout() == Layout::ROW_MAJOR, "This op only supports row major tensors.");
        TT_FATAL(input_tensor.get_shape()[dim] % num_splits == 0, "Split dimension must be divisible by num_splits.");
        auto input_shape = input_tensor.get_shape();
        auto input_rank = input_shape.size();

        const bool on_host = input_tensor.storage_type() == StorageType::OWNED || input_tensor.storage_type() == StorageType::BORROWED;
        std::optional<Device *> dev = on_host ? std::nullopt : std::make_optional(input_tensor.device());

        Tensor preprocessed = Tensor(input_tensor);
        preprocessed = ttnn::unsqueeze_to_4D(preprocessed); // ensure we're 4D before slicing
        dim += 4 - input_rank; // convert to 4D index

        if (!on_host && input_tensor.get_dtype() == DataType::BFLOAT16) {
            preprocessed = preprocessed.cpu(); // bf16 tensors must be handled on host due to limitations in slice
        }

        auto preproc_shape = preprocessed.get_shape();

        auto chunk_len = preproc_shape[dim] / num_splits;

        std::vector<Tensor> output_tensors;
        output_tensors.reserve(num_splits);

        for (int i = 0; i < num_splits; i++) {
            auto start = i*chunk_len;
            auto end = start + chunk_len - 1;

            std::vector<uint32_t> start_shape(preproc_shape.size(), 0);
            start_shape[dim] = start;

            std::vector<uint32_t> end_shape(preproc_shape.size());
            for (int j = 0; j < end_shape.size(); j++) {
                if (j == dim) {
                    end_shape[j] = end;
                } else {
                    end_shape[j] = preproc_shape[j] - 1;
                }
            }

            Tensor output_chunk = ttnn::slice(preprocessed,
                                              tt::tt_metal::LegacyShape(start_shape),
                                              tt::tt_metal::LegacyShape(end_shape),
                                              std::nullopt,
                                              mem_config);
            if (input_rank < 4) {
                output_chunk = ttnn::squeeze_from_4D(output_chunk, input_rank);
            }

            tt::tt_metal::Layout layout = input_tensor.get_layout();
            if (dev && (input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::UINT16)
                && chunk_len % 2 != 0) {
                layout = Layout::TILE; // bf16 and uint16 tensors must be tiled if the chunk length is odd due to packing constraints
                output_chunk = output_chunk.pad_to_tile(0.0);
            }

            output_chunk = output_chunk.to(layout);

            if (dev) {
                output_chunk = output_chunk.to(*dev);
            }

            output_tensors.push_back(output_chunk);
        }

        return output_tensors;
    }

    std::vector<Tensor> impl_split_last_dim_two_chunks_tiled(const Tensor &input_tensor, const MemoryConfig &mem_config) {
        auto input_shape = input_tensor.get_legacy_shape();
        auto padded_input_shape = ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(input_shape);
        ttnn::operations::experimental::auto_format::FormatParams input_format_params = {.pad_shape = padded_input_shape, .pad_value = 0.0, .target_layout = Layout::TILE};
        return operation::run_with_autoformat(SplitDeviceOperation{2, 3, mem_config}, {input_tensor}, {input_format_params}, {Layout::TILE, Layout::TILE});
    }

    std::vector<Tensor> split_last_dim_two_chunks_tiled(const Tensor &input_tensor, const MemoryConfig &mem_config) {
        const auto shape = input_tensor.get_legacy_shape();
        const bool pre_post_reshape = shape[0] > 1;

        if (!pre_post_reshape) {
            return impl_split_last_dim_two_chunks_tiled(input_tensor, mem_config);
        }

        const int W = 1, Z = shape[0] * shape[1], Y = shape[2], X = shape[3];
        const Tensor &reshaped_tensor = ttnn::reshape_on_device(input_tensor, 1, -1, Y, X, mem_config);

        auto part_reshaped = impl_split_last_dim_two_chunks_tiled(reshaped_tensor, mem_config);

        std::vector<Tensor> results;
        results.reserve(part_reshaped.size());
        for (auto &part : part_reshaped) results.emplace_back(ttnn::reshape_on_device(part, -1, shape[1], Y, X / 2, mem_config));

        return results;
    }


std::vector<Tensor> split_dim_n_chunks_tiled(
    const Tensor &input_tensor, int dim /* = 3 */, int num_splits, const MemoryConfig &mem_config /* = default */) {
    TT_FATAL(num_splits == 2, "ttnn.split currently only supports split in 2 in tiled layout, but {} is passed", num_splits);
    if (dim == 3) {
        return split_last_dim_two_chunks_tiled(input_tensor, mem_config);
    }
    Tensor ref_input_tensor = ttnn::transpose(input_tensor, dim, 3, mem_config);
    auto transposed_result = split_last_dim_two_chunks_tiled(ref_input_tensor, mem_config);
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

    if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
        return detail::split_dim_n_chunks_rm(input_tensor, dim, num_splits,  memory_config);
    } else {
        return detail::split_dim_n_chunks_tiled(input_tensor, dim, num_splits, memory_config);
    }
}

std::vector<ttnn::Tensor> SplitOperation::invoke(
    const ttnn::Tensor& input_tensor,
    int64_t& num_splits,
    int64_t& dim,
    const std::optional<MemoryConfig>& memory_config) {
    return invoke(DefaultQueueId, input_tensor, num_splits, dim, memory_config);
}

std::vector<ttnn::Tensor> SplitOperation::invoke(const ttnn::Tensor& input_tensor, int64_t& num_splits,  int64_t& dim) {
    return invoke(DefaultQueueId, input_tensor, num_splits, dim, std::nullopt);
}

} // ttnn::operations::data_movement namespace
