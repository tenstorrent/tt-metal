// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/binary_ng/types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
namespace ttnn::operations::binary_ng {

enum class SubtileBroadcastType {
    NONE,         // both tensors have equal tile dimensions (H & W)
    SCALAR_A,     // a is a scalar (H = 1, W = 1)
    SCALAR_B,     // b is a scalar (H = 1, W = 1)
    ROW_A_COL_B,  // a has a single tile row, b has a single tile column
    ROW_B_COL_A,  // b has a single tile row, a has a single tile column
    ROW_A,        // a has a single tile row, b is full
    ROW_B,        // b has a single tile row, a is full
    COL_A,        // a has a single tile column, b is full
    COL_B,        // b has a single tile column, a is full
};

SubtileBroadcastType get_subtile_broadcast_type(uint32_t a_h, uint32_t a_w, uint32_t b_h, uint32_t b_w);

struct BinaryNgDeviceOperation {
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct operation_attributes_t {
        BinaryOpType binary_op_type;
        ttnn::SmallVector<unary::UnaryWithParam> lhs_activations;
        ttnn::SmallVector<unary::UnaryWithParam> rhs_activations;
        ttnn::SmallVector<unary::UnaryWithParam> post_activations;
        std::optional<float> scalar;
        tt::tt_metal::MemoryConfig memory_config;
        DataType input_dtype;
        std::optional<DataType> dtype;
        const CoreRangeSet worker_grid;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config;
        SubtileBroadcastType subtile_broadcast_type = SubtileBroadcastType::NONE;
        bool is_sfpu = false;
        bool is_quant_op = false;

        DataType get_dtype() const;
    };

    struct tensor_args_t {
        const Tensor& input_tensor_a;
        std::optional<Tensor> input_tensor_b;
        std::optional<Tensor> output_tensor;
    };

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    static tt::tt_metal::ProgramDescriptor create_program(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void validate(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);

    // tensor-tensor invocation
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor_a_arg,
        const Tensor& input_tensor_b_arg,
        BinaryOpType binary_op_type,
        const std::optional<const DataType>& output_dtype,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<Tensor>& optional_output_tensor,
        tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
        tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
        tt::stl::Span<const unary::UnaryWithParam> post_activations);

    // tensor-scalar invocation
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor_a_arg,
        float scalar,
        BinaryOpType binary_op_type,
        const std::optional<const DataType>& output_dtype,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<Tensor>& optional_output_tensor,
        tt::stl::Span<const unary::UnaryWithParam> lhs_activations,
        tt::stl::Span<const unary::UnaryWithParam> rhs_activations,
        tt::stl::Span<const unary::UnaryWithParam> post_activations);
};

}  // namespace ttnn::operations::binary_ng

namespace ttnn::prim {
constexpr auto binary_ng =
    ttnn::register_operation<"ttnn::prim::binary_ng", ttnn::operations::binary_ng::BinaryNgDeviceOperation>();
}  // namespace ttnn::prim
