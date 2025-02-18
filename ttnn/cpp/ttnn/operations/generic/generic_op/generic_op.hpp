// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn::operations::generic {

struct operation_attributes_t;

struct GenericOp {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        const operation_attributes_t& operation_attributes,
        const std::vector<Tensor>& io_tensors = {});

    static Tensor invoke(
        const Tensor& input_tensor,
        const operation_attributes_t& operation_attributes,
        const std::vector<Tensor>& io_tensors = {});
};  // struct GenericOp

}   // namespace ttnn::operations::generic

namespace ttnn {
constexpr auto generic_op = ttnn::register_operation<"ttnn::generic_op", ttnn::operations::generic::GenericOp>();
}  // namespace ttnn
