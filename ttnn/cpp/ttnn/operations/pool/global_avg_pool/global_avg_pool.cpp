// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/global_avg_pool/global_avg_pool.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/reshape/reshape.hpp"

namespace tt {
namespace tt_metal {

template <PoolType pool>
Tensor pool_2d(const Tensor& input, const MemoryConfig& memory_config, const std::optional<DataType>& output_dtype) {
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input tensor needs to be on device");
    auto input_shape = input.get_legacy_shape();
    switch (pool) {
        case PoolType::AVG: {
            uint32_t height_without_padding = input.get_logical_shape()[-2];
            return ttnn::sum(
                input,
                int(input_shape.rank() - 2),
                true,
                memory_config,
                std::nullopt,
                1 / float(height_without_padding));
        }
        default: TT_THROW("Undefined pool type");
    }
}

Tensor global_avg_pool2d(
    const Tensor& input, const MemoryConfig& memory_config, const std::optional<DataType>& output_dtype) {
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input tensor needs to be on device");
    auto output = input;

    tt::tt_metal::LegacyShape in_shape = input.get_legacy_shape();
    auto input_padding = in_shape.padding();
    TT_FATAL(input_padding[1].front == 0 and input_padding[1].back == 0, "Padding along second dim is not supported");
    auto output_padding = Padding(
        {input_padding[0], {0, 0}, {0, input_padding[2].back * in_shape[1]}, input_padding[3]},
        input_padding.pad_value());
    auto output_shape =
        tt::tt_metal::LegacyShape({in_shape[0], 1, in_shape[1] * in_shape[2], in_shape[3]}, output_padding);
    output = ttnn::experimental::view(output, output_shape);

    output = pool_2d<PoolType::AVG>(output, memory_config, output_dtype);
    return output;
}

}  // namespace tt_metal
}  // namespace tt
