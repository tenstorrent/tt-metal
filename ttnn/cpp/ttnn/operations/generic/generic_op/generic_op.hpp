// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <unordered_map>

namespace ttnn::operations::generic {

enum CBIndex : std::uint8_t;
struct circular_buffer_attributes_t;
struct data_movement_attributes_t;
struct compute_attributes_t;

struct GenericOp {
    static Tensor invoke(
        QueueId queue_id,
        const std::vector<ttnn::Tensor>& input_tensors,
        const std::unordered_map<tt::CBIndex, circular_buffer_attributes_t>&,
        const std::vector<compute_attributes_t>&,
        const std::vector<data_movement_attributes_t>&,
        const std::vector<Tensor>& io_tensors = {});

    static Tensor invoke(
        const std::vector<ttnn::Tensor>& input_tensors,
        const std::unordered_map<tt::CBIndex, circular_buffer_attributes_t>&,
        const std::vector<compute_attributes_t>&,
        const std::vector<data_movement_attributes_t>&,
        const std::vector<Tensor>& io_tensors = {});
};  // struct GenericOp

}   // namespace ttnn::operations::generic

namespace ttnn {
constexpr auto generic_op = ttnn::register_operation<"ttnn::generic_op", ttnn::operations::generic::GenericOp>();
}  // namespace ttnn
