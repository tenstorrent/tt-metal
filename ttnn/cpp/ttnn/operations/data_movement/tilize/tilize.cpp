// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize.hpp"

#include "device/tilize_op.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {
using OwnedTilizeArgs = std::tuple<ttnn::Tensor>;
using BaseTilizeType = std::function<ttnn::Tensor(const ttnn::Tensor&)>;

using MassagedTilize = MassagedOperation<ttnn::Tensor, const ttnn::Tensor&>;
using MassagedTilizeParams = MassagedOperationParams<ttnn::Tensor, const ttnn::Tensor&>;

MassagedTilize build_ndiml_tilize(BaseTilizeType base_tilize) {
    auto original_shape = std::make_shared<ttnn::Shape>(ttnn::Shape{});
    return MassagedTilize(MassagedTilizeParams{
        .predicate = [](const ttnn::Tensor& input_tensor) -> bool { return input_tensor.get_shape().rank() > 4; },
        .pre_transform = [=](const ttnn::Tensor& input_tensor) -> OwnedTilizeArgs {
            *original_shape = input_tensor.get_shape();
            ttnn::Tensor squeezed_tensor = squeeze_from_ND_to_4D(input_tensor);
            return std::make_tuple(squeezed_tensor);
        },
        .post_transform = [=](const ttnn::Tensor& output) -> ttnn::Tensor {
            auto unsqueezed_tensor = ttnn::reshape(output, *original_shape);
            return unsqueezed_tensor;
        },
        .operation = std::move(base_tilize)});
}

ttnn::Tensor ExecuteTilize::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    auto base_tilize = [=](const ttnn::Tensor& input_tensor) {
        return operation::run(
            Tilize{
                memory_config.value_or(input_tensor.memory_config()),
                output_dtype.value_or(input_tensor.get_dtype()),
                use_multicore},
            {input_tensor},
            {},
            {},
            queue_id)[0];
    };

    return build_ndiml_tilize(base_tilize)(input_tensor);
}

ttnn::Tensor ExecuteTilize::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore) {
    return invoke(DefaultQueueId, input_tensor, memory_config, output_dtype, use_multicore);
}

}  // namespace ttnn::operations::data_movement
