// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <unordered_map>
#include "generic_op_types.hpp"

namespace ttnn::operations::generic {

enum CBIndex : std::uint8_t;

struct GenericOp {
    static Tensor invoke(
        const std::vector<Tensor>& io_tensors,
        const program_attributes_t& program_attributes
    );
};  // struct GenericOp

}   // namespace ttnn::operations::generic

namespace ttnn {
constexpr auto generic_op = 
    ttnn::register_operation_with_auto_launch_op<"ttnn::generic_op", ttnn::operations::generic::GenericOp>();
}  // namespace ttnn
