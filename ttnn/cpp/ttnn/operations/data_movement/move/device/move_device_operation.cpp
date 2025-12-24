// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "move_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::prim {
ttnn::operations::data_movement::move::MoveDeviceOperation::tensor_return_value_t move(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const MoveOpParallelizationStrategy& move_op_parallelization_strategy) {
    using OperationType = ttnn::operations::data_movement::move::MoveDeviceOperation;
    bool backwards = false;
    if (move_op_parallelization_strategy == MoveOpParallelizationStrategy::MULTI_CORE) {
        Buffer* src_buffer = input_tensor.buffer();
        Buffer* dst_buffer = output_tensor.buffer();
        const bool src_and_dst_in_l1 = src_buffer->buffer_type() == tt::tt_metal::BufferType::L1 &&
                                       dst_buffer->buffer_type() == tt::tt_metal::BufferType::L1;
        const uint32_t src_base = src_buffer->address();
        const uint32_t dst_base = dst_buffer->address();
        const uint32_t copy_size_bytes = dst_buffer->size();
        const bool ranges_overlap = (src_base < dst_base + copy_size_bytes) && (dst_base < src_base + copy_size_bytes);
        backwards = src_and_dst_in_l1 && ranges_overlap && (dst_base > src_base);
    }
    return ttnn::device_operation::detail::launch_on_device<OperationType>(
        OperationType::operation_attributes_t{
            .output_mem_config = output_mem_config,
            .move_op_parallelization_strategy = move_op_parallelization_strategy,
            .backwards = backwards},
        OperationType::tensor_args_t{.input_tensor = input_tensor, .output_tensor = output_tensor});
}
}  // namespace ttnn::prim

namespace ttnn::operations::data_movement::move {

}  // namespace ttnn::operations::data_movement::move
