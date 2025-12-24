// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/operations/data_movement/repeat/device/repeat_program_factory_last_dim.hpp"
#include "ttnn/operations/data_movement/repeat/device/repeat_program_factory_higher_dim.hpp"
#include "ttnn/operations/data_movement/repeat/device/repeat_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::prim {
ttnn::operations::data_movement::repeat::RepeatDeviceOperation::tensor_return_value_t repeat(
    const Tensor& input,
    uint32_t m_num_repeats,
    bool m_is_last_dim,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = ttnn::operations::data_movement::repeat::RepeatDeviceOperation;
    return ttnn::device_operation::detail::launch_on_device<OperationType>(
        OperationType::operation_attributes_t{
            .m_num_repeats = m_num_repeats, .m_is_last_dim = m_is_last_dim, .m_output_mem_config = output_mem_config},
        OperationType::tensor_args_t{.input = input});
}
}  // namespace ttnn::prim

namespace ttnn::operations::data_movement::repeat {
}  // namespace ttnn::operations::data_movement::repeat
