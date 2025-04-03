// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/core_coord.hpp>
#include <optional>
#include <string>

#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/tile.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/global_cb_utils.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
class Shape;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn {
namespace operations {
namespace unary {
enum class UnaryOpType;
struct UnaryWithParam;
}  // namespace unary
}  // namespace operations

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

namespace operations {
namespace matmul {

namespace detail {

bool is_input_batched(const ttnn::Shape& logical_shape);

}  // namespace detail

std::optional<UnaryWithParam> get_fused_activation(const std::optional<const std::string>& activation);

ttnn::Tensor bound_matmul(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const ttnn::Tensor>& bias,
    const struct Matmul& parameters,
    const uint8_t& queue_id,
    std::optional<ttnn::Tensor>& optional_output_tensor);

struct MatmulOperation {
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const bool transpose_a = false,
        const bool transpose_b = false,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const MatmulProgramConfig>& program_config = std::nullopt,
        const std::optional<const std::string>& activation = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<const CoreGrid> core_grid = std::nullopt,
        const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        const std::optional<const tt::tt_metal::DeviceGlobalCircularBuffer>& global_cb = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt);
};

struct LinearOperation {
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<const Tensor>& bias = std::nullopt,
        const bool transpose_a = false,
        const bool transpose_b = false,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const MatmulProgramConfig>& program_config = std::nullopt,
        const std::optional<const std::string>& activation = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<const CoreGrid> core_grid = std::nullopt,
        const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        const std::optional<const tt::tt_metal::DeviceGlobalCircularBuffer>& global_cb = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt);
};

}  // namespace matmul
}  // namespace operations
constexpr auto matmul = ttnn::register_operation<"ttnn::matmul", operations::matmul::MatmulOperation>();
constexpr auto linear = ttnn::register_operation<"ttnn::linear", operations::matmul::LinearOperation>();
}  // namespace ttnn
