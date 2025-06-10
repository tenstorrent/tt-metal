// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/global_avg_pool/global_avg_pool.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "cpp/ttnn/operations/experimental/reshape/view.hpp"

namespace tt {
namespace tt_metal {

template <PoolType pool>
Tensor pool_2d(const Tensor& input, const MemoryConfig& memory_config, const std::optional<DataType>& output_dtype) {
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input tensor needs to be on device");
    auto input_shape = input.padded_shape();
    switch (pool) {
        case PoolType::AVG: {
            uint32_t height_without_padding = input.logical_shape()[-2];
            return ttnn::operations::reduction::pool_sum(
                input, int(input_shape.rank() - 2), memory_config, std::nullopt, 1 / float(height_without_padding));
        }
        default: TT_THROW("Undefined pool type");
    }
}

Tensor global_avg_pool2d(
    const Tensor& input, const MemoryConfig& memory_config, const std::optional<DataType>& output_dtype) {
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input tensor needs to be on device");
    auto output = input;

    auto in_shape = input.padded_shape();
    ttnn::Shape output_shape({in_shape[0], 1, in_shape[1] * in_shape[2], in_shape[3]});
    output = ttnn::experimental::view(output, output_shape);

    output = pool_2d<PoolType::AVG>(output, memory_config, output_dtype);
    return output;
}

}  // namespace tt_metal
}  // namespace tt
