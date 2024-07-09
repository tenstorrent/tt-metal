// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/cpp/ttnn/operations/core.hpp"


#include "tt_eager/tt_dnn/op_library/run_operation.hpp"
#include "device/slice_op.hpp"

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

        return operation::run(
                   Slice{output_tensor_start, output_tensor_end, memory_config},
                   {input_tensor}, {}, {}, queue_id)
            .at(0);

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


    static ttnn::Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Shape output_tensor_start,
        const ttnn::Shape output_tensor_end,
        const std::optional<MemoryConfig>& memory_config_arg) {

        std::vector<uint32_t> output_tensor_start_vec(output_tensor_start.rank());
        std::vector<uint32_t> output_tensor_end_vec(output_tensor_end.rank());

        for(uint32_t dim = 0; dim<output_tensor_start.rank(); dim++) {
            output_tensor_start_vec[dim] = output_tensor_start[dim];
        }
        
        for(uint32_t dim = 0; dim<output_tensor_end.rank(); dim++) {
            output_tensor_end_vec[dim] = output_tensor_end[dim];
        }

        return execute_on_worker_thread(
                queue_id, 
                input_tensor, 
                output_tensor_start_vec, 
                output_tensor_end_vec, 
                memory_config_arg
                );


    }


};

}  // namespace data_movement
}  // namespace operations

constexpr auto slice = ttnn::register_operation<ttnn::operations::data_movement::ExecuteSlice>("ttnn::slice");

}  // namespace ttnn
