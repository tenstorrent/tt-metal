// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <magic_enum/magic_enum.hpp>
#include <optional>
#include <variant>

#include <tt-metalium/command_queue.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::binary {

struct BinaryDeviceOperation {
    struct operation_attributes_t {
        BinaryOpType binary_op_type;
        const std::optional<unary::FusedActivations> activations;
        const std::optional<unary::UnaryWithParam> input_tensor_a_activation;
        const std::optional<float> scalar;
        const MemoryConfig memory_config;
        const DataType dtype;
        const CoreRangeSet worker_grid;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config;
    };
    struct tensor_args_t {
        const Tensor& input_tensor_a;
        std::optional<Tensor> input_tensor_b;
        std::optional<Tensor> output_tensor;
    };
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ElementWiseMultiCore {
        static tt::tt_metal::ProgramDescriptor create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    struct ElementWiseMultiCoreSfpu {
        static tt::tt_metal::ProgramDescriptor create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };
    struct BroadcastWidthMultiCore {
        static tt::tt_metal::ProgramDescriptor create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    struct BroadcastHeightMultiCore {
        static tt::tt_metal::ProgramDescriptor create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    struct BroadcastHeightAndWidthMultiCore {
        static tt::tt_metal::ProgramDescriptor create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    struct BroadcastHeightMultiCoreSharded {
        static tt::tt_metal::ProgramDescriptor create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    struct BroadcastHeightMultiCoreShardedOptimized {
        static tt::tt_metal::ProgramDescriptor create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    static tt::tt_metal::ProgramDescriptor create_program(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void validate(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModel create_op_performance_model(
        const operation_attributes_t& attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor_a_arg,
        const Tensor& input_tensor_b_arg,
        BinaryOpType binary_op_type,
        const std::optional<const DataType>& output_dtype,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> optional_output_tensor,
        std::optional<unary::FusedActivations> activations,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor_a_arg,
        float scalar,
        BinaryOpType binary_op_type,
        const std::optional<const DataType>& output_dtype,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> optional_output_tensor,
        std::optional<unary::FusedActivations> activations,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation);

private:
    using program_factory_t = std::variant<
        ElementWiseMultiCore,
        ElementWiseMultiCoreSfpu,
        BroadcastWidthMultiCore,
        BroadcastHeightMultiCore,
        BroadcastHeightAndWidthMultiCore,
        BroadcastHeightMultiCoreSharded,
        BroadcastHeightMultiCoreShardedOptimized>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::binary

namespace ttnn::prim {
constexpr auto binary =
    ttnn::register_operation<"ttnn::prim::binary", ttnn::operations::binary::BinaryDeviceOperation>();
}  // namespace ttnn::prim
