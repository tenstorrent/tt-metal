// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/repeat/repeat_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/composite/composite_ops.hpp"
#include "ttnn/operations/core/core.hpp"

#include <ranges>

namespace ttnn {
namespace operations {
namespace data_movement {


struct Repeat {
    static ttnn::Tensor operator()(
        const ttnn::Tensor& input_tensor,
        const ttnn::Shape& shape,
        std::optional<MemoryConfig> output_mem_config = std::nullopt) {
        MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());
        auto output_tensor = tt::tt_metal::repeat(input_tensor, shape.value, mem_config);
        return output_tensor;
    }
};


}  // namespace data_movement
}  // namespace operations

constexpr auto repeat = ttnn::register_operation_with_auto_launch_op<"ttnn::repeat", ttnn::operations::data_movement::Repeat>();

}  // namespace ttnn
