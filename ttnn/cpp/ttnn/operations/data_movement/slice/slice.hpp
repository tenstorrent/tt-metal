// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/slice_op.hpp"
#include "tt_eager/tt_dnn/op_library/run_operation.hpp"
#include "ttnn/cpp/ttnn/operations/core.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

struct ExecuteSlice {
    static ttnn::Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        tt::tt_metal::Shape output_tensor_start,
        tt::tt_metal::Shape output_tensor_end,
        const std::optional<MemoryConfig>& memory_config_arg) {
        auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());

        // TODO: Generalize this early exit of slice for other cases
        auto& input_tensor_shape = input_tensor.get_legacy_shape();
        if (input_tensor.is_sharded() && input_tensor.memory_config() == memory_config &&
            input_tensor_shape.rank() > 1 && input_tensor_shape.rank() == output_tensor_start.rank() &&
            output_tensor_start.rank() == output_tensor_end.rank()) {
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
                auto padded_shape = ttnn::Shape(tt::tt_metal::Shape(new_shape, new_pad));
                return Tensor(input_tensor.storage(), padded_shape, input_tensor.dtype(), input_tensor.layout());
            }
        }

        return operation::run(
                   Slice{output_tensor_start, output_tensor_end, memory_config}, {input_tensor}, {}, {}, queue_id)
            .at(0);
    }

    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        tt::tt_metal::Shape output_tensor_start,
        tt::tt_metal::Shape output_tensor_end,
        const std::optional<MemoryConfig>& memory_config_arg) {
        return execute_on_worker_thread(0, input_tensor, output_tensor_start, output_tensor_end, memory_config_arg);
    }

    static ttnn::Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        std::vector<uint32_t> output_tensor_start,
        std::vector<uint32_t> output_tensor_end,
        const std::optional<MemoryConfig>& memory_config_arg) {
        return execute_on_worker_thread(
            queue_id,
            input_tensor,
            tt::tt_metal::Shape(output_tensor_start),
            tt::tt_metal::Shape(output_tensor_end),
            memory_config_arg);
    }

    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        std::vector<uint32_t> output_tensor_start,
        std::vector<uint32_t> output_tensor_end,
        const std::optional<MemoryConfig>& memory_config_arg) {
        return execute_on_worker_thread(0, input_tensor, output_tensor_start, output_tensor_end, memory_config_arg);
    }

    static ttnn::Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Shape output_tensor_start,
        const ttnn::Shape output_tensor_end,
        const std::optional<MemoryConfig>& memory_config_arg) {
        std::vector<uint32_t> output_tensor_start_vec(output_tensor_start.rank());
        std::vector<uint32_t> output_tensor_end_vec(output_tensor_end.rank());

        for (uint32_t dim = 0; dim < output_tensor_start.rank(); dim++) {
            output_tensor_start_vec[dim] = output_tensor_start[dim];
        }

        for (uint32_t dim = 0; dim < output_tensor_end.rank(); dim++) {
            output_tensor_end_vec[dim] = output_tensor_end[dim];
        }

        return execute_on_worker_thread(
            queue_id, input_tensor, output_tensor_start_vec, output_tensor_end_vec, memory_config_arg);
    }
};

}  // namespace data_movement
}  // namespace operations

constexpr auto slice = ttnn::register_operation<ttnn::operations::data_movement::ExecuteSlice>("ttnn::slice");

}  // namespace ttnn
