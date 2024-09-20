// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/common/constants.hpp"
#include "slice.hpp"
#include "device/slice_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/common/constants.hpp"


namespace ttnn::operations::data_movement {

ttnn::Tensor SliceOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    tt::tt_metal::LegacyShape output_tensor_start,
    tt::tt_metal::LegacyShape output_tensor_end,
    const std::optional<tt::tt_metal::LegacyShape> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {
    std::optional<tt::tt_metal::LegacyShape> modified_step = step;
    if (modified_step.has_value()) {
        if (std::all_of(modified_step->begin(), modified_step->end(), [](int32_t s) { return s == 1; })) {
            modified_step = std::nullopt;
        }
    }
    if (input_tensor.storage_type() != StorageType::DEVICE) {
        TT_FATAL(!modified_step.has_value(), "Host tensor slice does not support strides");
        tt::tt_metal::LegacyShape output_tensor_shape = {
            output_tensor_end[0] - output_tensor_start[0] + 1,
            output_tensor_end[1] - output_tensor_start[1] + 1,
            output_tensor_end[2] - output_tensor_start[2] + 1,
            output_tensor_end[3] - output_tensor_start[3] + 1,
        };
        // if we support negative strides, we can't do this early exit
        if (input_tensor.get_legacy_shape() == output_tensor_shape) {
            return input_tensor;
        } else {
            return input_tensor.unpad(output_tensor_start, output_tensor_end);
        }
    }
    else {
        auto memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config() : memory_config_arg.value_or(input_tensor.memory_config());
        // TODO: Generalize this early exit of slice for other cases
        auto& input_tensor_shape = input_tensor.get_legacy_shape();
        if (input_tensor.is_sharded() && input_tensor.memory_config() == memory_config &&
            input_tensor_shape.rank() > 1 && input_tensor_shape.rank() == output_tensor_start.rank() &&
            output_tensor_start.rank() == output_tensor_end.rank()) {
            TT_FATAL(!modified_step.has_value(), "Sharded tensor slice implementation does not support striding");
            uint32_t i;
            // Require all leading dims to be 1 (TODO: This can be relaxed to support outermost non-1 dim unpadding)
            bool in_place_unpad = true;
            for (i = 0; i < input_tensor.get_legacy_shape().rank() - 2; ++i) {
                in_place_unpad &=
                    output_tensor_start[i] == 0 && output_tensor_end[i] == 0 && input_tensor_shape[i] == 1;
            }
            in_place_unpad &= output_tensor_start[i] == 0 &&
                              tt::div_up(output_tensor_end[i] + 1, input_tensor.shard_spec().value().shape[0]) ==
                                  tt::div_up(input_tensor_shape[i], input_tensor.shard_spec().value().shape[0]);
            i++;
            in_place_unpad &= output_tensor_start[i] == 0 && output_tensor_end[i] == input_tensor_shape[i] - 1;
            if (in_place_unpad) {
                auto new_shape = input_tensor.get_legacy_shape();
                auto new_pad = new_shape.padding();

                std::size_t unpad_val = input_tensor_shape[-2] - output_tensor_end[-2] - 1;
                new_shape[-2] -= unpad_val;
                new_pad[-2].back -= std::min(unpad_val, new_pad[-2].back);
                auto padded_shape = ttnn::Shape(tt::tt_metal::LegacyShape(new_shape, new_pad));
                return Tensor(input_tensor.storage(), padded_shape, input_tensor.dtype(), input_tensor.layout());
            }
        }

        return operation::run(
                   SliceDeviceOperation{output_tensor_start, output_tensor_end, modified_step, memory_config}, {input_tensor}, {}, {optional_output_tensor}, queue_id)
            .at(0);

    }
}

ttnn::Tensor SliceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    tt::tt_metal::LegacyShape output_tensor_start,
    tt::tt_metal::LegacyShape output_tensor_end,
    const std::optional<tt::tt_metal::LegacyShape> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {
    return invoke(DefaultQueueId, input_tensor, output_tensor_start, output_tensor_end, step, memory_config_arg, optional_output_tensor);
}

ttnn::Tensor SliceOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    tt::tt_metal::Array1D output_tensor_start,
    tt::tt_metal::Array1D output_tensor_end,
    const std::optional<tt::tt_metal::Array1D> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {
    return invoke(
        queue_id,
        input_tensor,
        tt::tt_metal::LegacyShape(output_tensor_start),
        tt::tt_metal::LegacyShape(output_tensor_end),
        step.has_value() ? std::optional<tt::tt_metal::LegacyShape>(tt::tt_metal::LegacyShape(step.value())) : std::nullopt,
        memory_config_arg,
        optional_output_tensor);
}

ttnn::Tensor SliceOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    tt::tt_metal::Array4D output_tensor_start,
    tt::tt_metal::Array4D output_tensor_end,
    const std::optional<tt::tt_metal::Array4D> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {
    return invoke(
        queue_id,
        input_tensor,
        tt::tt_metal::LegacyShape(output_tensor_start),
        tt::tt_metal::LegacyShape(output_tensor_end),
        step.has_value() ? std::optional<tt::tt_metal::LegacyShape>(tt::tt_metal::LegacyShape(step.value())) : std::nullopt,
        memory_config_arg,
        optional_output_tensor);
}

ttnn::Tensor SliceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    tt::tt_metal::Array4D output_tensor_start,
    tt::tt_metal::Array4D output_tensor_end,
    const std::optional<tt::tt_metal::Array4D> step,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor) {
    return invoke(DefaultQueueId, input_tensor, output_tensor_start, output_tensor_end, step, memory_config_arg, optional_output_tensor);
}

ttnn::Tensor SliceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    tt::tt_metal::Array4D output_tensor_start,
    tt::tt_metal::Array4D output_tensor_end,
    const std::optional<tt::tt_metal::Array4D> step) {
    return invoke(DefaultQueueId, input_tensor, output_tensor_start, output_tensor_end, step, std::nullopt, std::nullopt);
}

}  // namespace operations
