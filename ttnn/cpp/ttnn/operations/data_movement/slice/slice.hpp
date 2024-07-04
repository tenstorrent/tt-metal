// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tensor/types.hpp"
#include "ttnn/cpp/ttnn/operations/core.hpp"
#include "tt_eager/tt_dnn/op_library/unpad/unpad_op.hpp"

#include <ranges>


namespace ttnn {
namespace operations {
namespace data_movement {



struct ExecuteSlice {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            2,  // min rank
            4,  // max rank
            {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::uint16, ttnn::int32, ttnn::uint32},
            {ttnn::TILE_LAYOUT},
            true,   // can_be_on_device
            false,  // can_be_on_cpu
            false,  // can_be_scalar
            false   // is_optional}
        }};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const ttnn::Tensor& input_tensor, Args&&... args) {
        return std::make_tuple(input_tensor);
    }

    static ttnn::Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        tt::tt_metal::Shape output_tensor_start,
        tt::tt_metal::Shape output_tensor_end,
        const std::optional<MemoryConfig>& memory_config_arg) {

        auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());

        auto input_tensor_shape = input_tensor.get_legacy_shape();
        std::vector<uint32_t> output_tensor_shape = {
            output_tensor_end[0] - output_tensor_start[0] + 1,
            output_tensor_end[1] - output_tensor_start[1] + 1,
            output_tensor_end[2] - output_tensor_start[2] + 1,
            output_tensor_end[3] - output_tensor_start[3] + 1,
        };
        auto output_tensor = operation::run(
            tt::tt_metal::Unpad{
                .output_tensor_start=output_tensor_start,
                .output_tensor_end=output_tensor_end,
                .output_mem_config=memory_config,
                .output_shape=output_tensor_shape,
                .input_shape=input_tensor_shape
            },
            {input_tensor}).front();

        return output_tensor;
    }


    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        tt::tt_metal::Shape output_tensor_start,
        tt::tt_metal::Shape output_tensor_end,
        const std::optional<MemoryConfig>& memory_config_arg
        ) {

        
        return execute_on_worker_thread(
                0, 
                input_tensor, 
                output_tensor_start, 
                output_tensor_end, 
                memory_config_arg
                );
        
    }


    static ttnn::Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        std::vector<uint32_t> output_tensor_start,
        std::vector<uint32_t> output_tensor_end,
        const std::optional<MemoryConfig>& memory_config_arg) {

        return execute_on_worker_thread(queue_id, 
                input_tensor, 
                tt::tt_metal::Shape(output_tensor_start), 
                tt::tt_metal::Shape(output_tensor_end), 
                memory_config_arg
                );

    }

    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        std::vector<uint32_t> output_tensor_start,
        std::vector<uint32_t> output_tensor_end,
        const std::optional<MemoryConfig>& memory_config_arg) {

        return execute_on_worker_thread(
                0, 
                input_tensor, 
                output_tensor_start, 
                output_tensor_end, 
                memory_config_arg
                );
    }


};

}  // namespace data_movement
}  // namespace operations

constexpr auto slice = ttnn::register_operation<ttnn::operations::data_movement::ExecuteSlice>("ttnn::slice");

}  // namespace ttnn
