// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize.hpp"

#include "device/untilize_op.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {
using OwnedUntilizeArgs = std::tuple<ttnn::Tensor>;
using BaseUntilizeType = std::function<ttnn::Tensor(const ttnn::Tensor&)>;

using MassagedUntilize = MassagedOperation<ttnn::Tensor, const ttnn::Tensor&>;
using MassagedUntilizeParams = MassagedOperationParams<ttnn::Tensor, const ttnn::Tensor&>;

MassagedUntilize build_ndiml_untilize(BaseUntilizeType base_untilize) {
    auto original_shape = std::make_shared<ttnn::Shape>(ttnn::Shape{});
    return MassagedUntilize(MassagedUntilizeParams{
        .predicate = [](const ttnn::Tensor& input_tensor) -> bool { return input_tensor.get_shape().rank() > 4; },
        .pre_transform = [=](const ttnn::Tensor& input_tensor) -> OwnedUntilizeArgs {
            *original_shape = input_tensor.get_shape();
            ttnn::Tensor squeezed_tensor = squeeze_from_ND_to_4D(input_tensor);
            return std::make_tuple(squeezed_tensor);
        },
        .post_transform = [=](const ttnn::Tensor& output) -> ttnn::Tensor {
            auto unsqueezed_tensor = ttnn::reshape(output, *original_shape);
            return unsqueezed_tensor;
        },
        .operation = std::move(base_untilize)});
}

ttnn::Tensor ExecuteUntilize::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    bool use_multicore,
    bool use_pack_untilize,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    bool fp32_dest_acc_en =
        input_tensor.get_dtype() ==
        DataType::UINT32;  // MT: Currently only uint32 is moved to DST directly, fp32 is converted to fp16b

    auto base_untilize = [=](const ttnn::Tensor& input_tensor) {
        return operation::run(
            Untilize{
                memory_config.value_or(input_tensor.memory_config()),
                use_multicore,
                use_pack_untilize,
                fp32_dest_acc_en,
                sub_core_grids},
            {input_tensor},
            {},
            {},
            queue_id)[0];
    };

    return build_ndiml_untilize(base_untilize)(input_tensor);
}

ttnn::Tensor ExecuteUntilize::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    bool use_multicore,
    bool use_pack_untilize,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return invoke(DefaultQueueId, input_tensor, memory_config, use_multicore, use_pack_untilize, sub_core_grids);
}

}  // namespace ttnn::operations::data_movement
