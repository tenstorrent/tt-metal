// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/bcast/bcast.hpp"

#include "ttnn/operations/data_movement/bcast/device/bcast_device_operation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn::operations::data_movement {

// Does a broadcast
Tensor BcastOperation::invoke(
    uint8_t queue_id,
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    BcastOpMath bcast_op,
    BcastOpDim bcast_dim,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<Tensor> output_tensor) {
    auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor_a}))};

    operation::launch_with_autoformat(
        [bcast_op, bcast_dim, output_memory_config, output_tensor, queue_id](
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors,
            const std::vector<std::optional<Tensor>> &optional_output_tensors) mutable -> std::vector<Tensor> {
            using tt::constants::TILE_HEIGHT;
            using tt::constants::TILE_WIDTH;
            auto &input_tensor_a = input_tensors.at(0);
            auto &input_tensor_b = input_tensors.at(1);
            if (bcast_dim == BcastOpDim::W) {
                TT_FATAL(input_tensor_a.get_legacy_shape()[-2] == input_tensor_b.get_legacy_shape()[-2], "Error");
                if (input_tensor_b.get_layout() == Layout::TILE) {
                    TT_FATAL(input_tensor_b.get_legacy_shape()[-1] == TILE_WIDTH, "Error");
                } else if (input_tensor_b.get_layout() == Layout::ROW_MAJOR) {
                    TT_FATAL(
                        input_tensor_b.get_legacy_shape()[-1] == 1 ||
                        input_tensor_b.get_legacy_shape()[-1] == TILE_WIDTH, "Error");
                } else {
                    TT_THROW("Unsupported layout");
                }
            } else if (bcast_dim == BcastOpDim::H) {
                TT_FATAL(input_tensor_a.get_legacy_shape()[-1] == input_tensor_b.get_legacy_shape()[-1], "Error");
                if (input_tensor_b.get_layout() == Layout::TILE) {
                    TT_FATAL(input_tensor_b.get_legacy_shape()[-2] == TILE_HEIGHT, "Error");
                } else if (input_tensor_b.get_layout() == Layout::ROW_MAJOR) {
                    TT_FATAL(
                        input_tensor_b.get_legacy_shape()[-2] == 1 ||
                        input_tensor_b.get_legacy_shape()[-2] == TILE_HEIGHT, "Error");
                } else {
                    TT_THROW("Unsupported layout");
                }
            } else if (bcast_dim == BcastOpDim::HW) {
                if (input_tensor_b.get_layout() == Layout::TILE) {
                    TT_FATAL(
                        input_tensor_b.get_legacy_shape()[-2] == TILE_HEIGHT &&
                        input_tensor_b.get_legacy_shape()[-1] == TILE_WIDTH, "Error");
                } else if (input_tensor_b.get_layout() == Layout::ROW_MAJOR) {
                    TT_FATAL(
                        (input_tensor_b.get_legacy_shape()[-2] == 1 && input_tensor_b.get_legacy_shape()[-1] == 1) ||
                        (input_tensor_b.get_legacy_shape()[-2] == TILE_HEIGHT &&
                         input_tensor_b.get_legacy_shape()[-1] == TILE_WIDTH), "Error");
                }
            }
            return operation::run_with_autoformat(
                EltwiseBinaryBroadcast{bcast_op, bcast_dim, output_memory_config},
                {input_tensor_a, input_tensor_b},
                {},
                {output_tensor},
                queue_id);
        },
        {input_tensor_a, input_tensor_b},
        output_tensors,
        {},
        {output_tensor});
    return output_tensors[0];
}

}  // namespace ttnn::operations::data_movement
