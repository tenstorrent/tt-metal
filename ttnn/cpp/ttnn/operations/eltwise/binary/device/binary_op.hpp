// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_eager/tensor/host_buffer/functions.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_eager/tt_dnn/op_library/compute_kernel_config.hpp"

#include "tt_eager/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_eager/tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::binary {

enum class BinaryOpType {
    ADD,
    SUB,
    MUL,
    GT,
    LT,
    LTE,
    GTE,
    EQ,
    NE,
    SQUARED_DIFFERENCE,
    BIAS_GELU,
    LOGADDEXP,
    LOGICAL_AND,
    LOGICAL_OR,
    LDEXP,
    LOGADDEXP2,
    DIV_FAST
};

using FusedActivations = std::vector<tt::tt_metal::UnaryWithParam>;
namespace utils {

std::map<string, string> get_defines(BinaryOpType op_type, const std::optional<DataType> in_dtype = std::nullopt, const std::optional<DataType> out_dtype = std::nullopt,
                                    const std::optional<FusedActivations> fused_activations = std::nullopt);

}  // namespace utils

constexpr uint8_t DefaultQueueId = 0;

struct Binary {
    BinaryOpType binary_op_type;
    bool in_place;
    const std::optional<std::vector<std::string>> activations;
    const MemoryConfig memory_config;
    const DataType dtype;
    std::optional<DeviceComputeKernelConfig> compute_kernel_config;

    void validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;

    operation::OpPerformanceModel create_op_performance_model(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "binary_op_type", "in_place", "activations", "memory_config", "dtype", "compute_kernel_config");
    const auto attribute_values() const {
        return std::forward_as_tuple(
            this->binary_op_type,
            this->in_place,
            this->activations,
            this->memory_config,
            this->dtype,
            this->compute_kernel_config);
    }
};

}  // namespace ttnn::operations::binary
