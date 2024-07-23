// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/avgpool/avg_pool.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/reduce/reduce_op.hpp"


namespace tt {
namespace tt_metal {

template<PoolType pool>
Tensor pool_2d(const Tensor& input, const MemoryConfig& memory_config, const std::optional<DataType>& output_dtype) {
    TT_ASSERT(input.storage_type() == StorageType::DEVICE, "Input tensor needs to be on device");
    auto input_shape = input.get_legacy_shape();
    switch (pool) {
        case PoolType::AVG: {
            auto height_without_padding = input.get_legacy_shape().without_padding()[-2];
            return reduce(input, ReduceOpMath::SUM, ReduceOpDim::H, 1 / float(height_without_padding), memory_config, output_dtype);
        }
        default:
            TT_ASSERT(false && "Undefined pool type");
    }
}

Tensor avg_pool2d(const Tensor& input, const MemoryConfig& memory_config, const std::optional<DataType>& output_dtype) {
    TT_ASSERT(input.storage_type() == StorageType::DEVICE, "Input tensor needs to be on device");
    auto output = input;

    Shape in_shape = input.get_legacy_shape();
    auto input_padding = in_shape.padding();
    TT_ASSERT(input_padding[1].front == 0 and input_padding[1].back == 0);
    auto output_padding = Padding({input_padding[0], {0, 0}, {0, input_padding[2].back * in_shape[1]}, input_padding[3]}, input_padding.pad_value());
    auto output_shape = Shape({in_shape[0], 1, in_shape[1] * in_shape[2], in_shape[3]}, output_padding);
    output = output.reshape(output_shape);

    output = pool_2d<PoolType::AVG>(output, memory_config, output_dtype);
    return output;
}

}  // namespace tt_metal
}  // namespace tt
