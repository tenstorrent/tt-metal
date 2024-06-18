
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/binary_backward_op.cpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement.hpp"
#include "tt_eager/tt_dnn/op_library/run_operation.hpp" //decorate_as_composite

namespace ttnn {

namespace operations::binary_backward {

// template <BinaryBackwardOpType binary_backward_op_type>
struct ExecuteBinaryBackward {
    static inline const std::array<TensorSchema, 3> input_tensor_schemas() {
        return {
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::uint16},
                {ttnn::TILE_LAYOUT},
                true,
                false,
                false,
                false},
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::uint16},
                {ttnn::TILE_LAYOUT},
                true,
                false,
                false,
                false},
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::uint16},
                {ttnn::TILE_LAYOUT},
                true,
                false,
                false,
                false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor &grad_tensor, const Tensor &input_tensor_a, const Tensor &input_tensor_b, Args &&...args) {
        return std::forward_as_tuple(grad_tensor, input_tensor_a, input_tensor_b);
    }

    //combination 1
    static std::vector<Tensor> execute_on_worker_thread(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const MemoryConfig &memory_config) {

        // auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());
        return tt::tt_metal::operation::decorate_as_composite(__func__, utils::_atan2_bw)(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, memory_config);
        }

    static inline std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {Tensor(operation::get_workers_for_op_output({input_tensor})),
                                            Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }
};

}  // operations::binary

// constexpr auto atan2_bw = ttnn::register_operation<operations::binary_backward::ExecuteBinaryBackward<operations::binary_backward::BinaryBackwardOpType::ATAN2_BW>>("ttnn::atan2_bw");
constexpr auto atan2_bw =
    ttnn::register_operation<ttnn::operations::binary_backward::ExecuteBinaryBackward>("ttnn::atan2_bw");


}  // namespace ttnn
