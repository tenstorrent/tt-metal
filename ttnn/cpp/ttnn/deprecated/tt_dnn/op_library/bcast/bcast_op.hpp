// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

enum class BcastOpMath { ADD, SUB, MUL };

enum class BcastOpDim { H, W, HW };

// TODO: Accept parallelization
enum class BcastOpParallelizationStrategy { MULTI_CORE_H_SHARDED, MULTI_CORE_H_SHARDED_OPTIMISED, MULTI_CORE_H, MULTI_CORE_W, MULTI_CORE_HW, SINGLE_CORE };

operation::ProgramWithCallbacks bcast_multi_core_h(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const Tensor &output_tensor,
    BcastOpMath bcast_op);
operation::ProgramWithCallbacks bcast_sharded_h(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const Tensor &output_tensor,
    BcastOpMath bcast_op);
operation::ProgramWithCallbacks bcast_sharded_h_optimised(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const Tensor &output_tensor,
    BcastOpMath bcast_op);
operation::ProgramWithCallbacks bcast_multi_core_w(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const Tensor &output_tensor,
    BcastOpMath bcast_op);
operation::ProgramWithCallbacks bcast_multi_core_hw(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const Tensor &output_tensor,
    BcastOpMath bcast_op,
    bool inplace);

struct EltwiseBinaryBroadcast {
    const BcastOpMath math_op;
    const BcastOpDim dim;
    const MemoryConfig output_mem_config;
    const bool in_place;

    void validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    BcastOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
};

inline Tensor run_bcast(
    uint8_t queue_id,
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    BcastOpMath bcast_op,
    BcastOpDim bcast_dim,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<Tensor> output_tensor = std::nullopt) {

    std::vector<Tensor> output_tensors = operation::launch_op(
        [bcast_op, bcast_dim, output_mem_config, output_tensor, queue_id] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            using tt::constants::TILE_HEIGHT;
            using tt::constants::TILE_WIDTH;
            auto& input_tensor_a = input_tensors.at(0);
            auto& input_tensor_b = input_tensors.at(1);
            if (bcast_dim == BcastOpDim::W) {
                TT_FATAL(input_tensor_a.get_legacy_shape()[-2] == input_tensor_b.get_legacy_shape()[-2]);
                if (input_tensor_b.get_layout() == Layout::TILE) {
                    TT_FATAL(input_tensor_b.get_legacy_shape()[-1] == TILE_WIDTH);
                } else if (input_tensor_b.get_layout() == Layout::ROW_MAJOR) {
                    TT_FATAL(input_tensor_b.get_legacy_shape()[-1] == 1 || input_tensor_b.get_legacy_shape()[-1] == TILE_WIDTH);
                } else {
                    TT_FATAL(false, "Unsupported layout");
                }
            } else if (bcast_dim == BcastOpDim::H) {
                TT_FATAL(input_tensor_a.get_legacy_shape()[-1] == input_tensor_b.get_legacy_shape()[-1]);
                if (input_tensor_b.get_layout() == Layout::TILE) {
                    TT_FATAL(input_tensor_b.get_legacy_shape()[-2] == TILE_HEIGHT);
                } else if (input_tensor_b.get_layout() == Layout::ROW_MAJOR) {
                    TT_FATAL(input_tensor_b.get_legacy_shape()[-2] == 1 || input_tensor_b.get_legacy_shape()[-2] == TILE_HEIGHT);
                } else {
                    TT_FATAL(false, "Unsupported layout");
                }
            } else if (bcast_dim == BcastOpDim::HW) {
                if (input_tensor_b.get_layout() == Layout::TILE) {
                    TT_FATAL(
                        input_tensor_b.get_legacy_shape()[-2] == TILE_HEIGHT &&
                        input_tensor_b.get_legacy_shape()[-1] == TILE_WIDTH);
                } else if (input_tensor_b.get_layout() == Layout::ROW_MAJOR) {
                    TT_FATAL(
                        (input_tensor_b.get_legacy_shape()[-2] == 1 && input_tensor_b.get_legacy_shape()[-1] == 1) ||
                        (input_tensor_b.get_legacy_shape()[-2] == TILE_HEIGHT &&
                        input_tensor_b.get_legacy_shape()[-1] == TILE_WIDTH));
                }
            }
            return operation::run_with_autoformat(
                    EltwiseBinaryBroadcast{bcast_op, bcast_dim, output_mem_config}, {input_tensor_a, input_tensor_b}, {}, {output_tensor}, queue_id);
        }, {input_tensor_a, input_tensor_b}, {}, {output_tensor});
    return output_tensors.at(0);
}

inline Tensor bcast(
    uint8_t queue_id,
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    BcastOpMath bcast_op,
    BcastOpDim bcast_dim,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<Tensor> output_tensor  = std::nullopt) {
       return run_bcast(queue_id, input_tensor_a, input_tensor_b, bcast_op, bcast_dim, output_mem_config, output_tensor);
    }
inline Tensor bcast(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    BcastOpMath bcast_op,
    BcastOpDim bcast_dim,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<Tensor> output_tensor  = std::nullopt) {
    uint8_t default_queue_id = 0;
       return run_bcast(default_queue_id, input_tensor_a, input_tensor_b, bcast_op, bcast_dim, output_mem_config, output_tensor);
    }

}  // namespace tt_metal

namespace operations {

namespace primary {

inline Tensor bcast(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    BcastOpMath bcast_op,
    BcastOpDim bcast_dim,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    bool in_place = false,
    std::optional<Tensor> output_tensor = std::nullopt,
    uint8_t queue_id = 0) {
    vector<Tensor> output = operation::run(EltwiseBinaryBroadcast{bcast_op, bcast_dim, mem_config, in_place}, {input_tensor_a, input_tensor_b}, {}, {output_tensor}, queue_id);
    if (in_place) {
        return input_tensor_a;
    } else if (output_tensor.has_value()){
        return output_tensor.value();
    } else {
        return output.at(0);
    }
}

}  // namespace primary

}  // namespace operations

}  // namespace tt

namespace bcast_op_utils {

std::map<std::string, std::string> get_defines(BcastOpDim bcast_dim, BcastOpMath bcast_math);

}  // namespace bcast_op_utils
