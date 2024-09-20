// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
namespace ttnn::operations::full_like {

struct FullLike {

    static Tensor invoke(
        const Tensor &input,
        const std::variant<float, int> fill_value,
        const std::optional<DataType> &dtype,
        const std::optional<Layout> &layout,
        const std::optional<MemoryConfig> &memory_config);

};
}

namespace ttnn {
constexpr auto full_like =
    ttnn::register_operation_with_auto_launch_op<"ttnn::full_like", ttnn::operations::full_like::FullLike>();
}
