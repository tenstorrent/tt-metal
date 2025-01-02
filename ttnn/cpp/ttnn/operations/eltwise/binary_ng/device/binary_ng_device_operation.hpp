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
        ttnn::SmallVector<unary::UnaryOpType> lhs_activations;
        ttnn::SmallVector<unary::UnaryOpType> rhs_activations;
        ttnn::SmallVector<unary::UnaryOpType> post_activations;
        std::optional<float> scalar;
        tt::tt_metal::MemoryConfig memory_config;
        DataType input_dtype;
        std::optional<DataType> dtype;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config;
        SubtileBroadcastType subtile_broadcast_type = SubtileBroadcastType::NONE;

        tt::stl::hash::hash_t to_hash() const;
        DataType get_dtype() const;
    };

    struct tensor_args_t {
        const Tensor& input_tensor_a;
        std::optional<Tensor> input_tensor_b;
        std::optional<Tensor> output_tensor;
    };

    struct ProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
            tt::tt_metal::KernelHandle compute_kernel_id;
            CoreCoord compute_with_storage_grid_size;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    // tensor-tensor invocation
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor_a_arg,
        const Tensor& input_tensor_b_arg,
        BinaryOpType binary_op_type,
        const std::optional<const DataType>& output_dtype,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> optional_output_tensor,
        tt::stl::Span<const unary::UnaryOpType> lhs_activations,
        tt::stl::Span<const unary::UnaryOpType> rhs_activations,
        tt::stl::Span<const unary::UnaryOpType> post_activations);

    // tensor-scalar invocation
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor_a_arg,
        float scalar,
        BinaryOpType binary_op_type,
        const std::optional<const DataType>& output_dtype,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> optional_output_tensor,
        tt::stl::Span<const unary::UnaryOpType> lhs_activations,
        tt::stl::Span<const unary::UnaryOpType> rhs_activations,
        tt::stl::Span<const unary::UnaryOpType> post_activations);

    // tensor-tensor inplace invocation
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor_a_arg,
        const Tensor& input_tensor_b_arg,
        BinaryOpType binary_op_type,
        const std::optional<const DataType>& output_dtype,
        tt::stl::Span<const unary::UnaryOpType> lhs_activations,
        tt::stl::Span<const unary::UnaryOpType> rhs_activations,
        tt::stl::Span<const unary::UnaryOpType> post_activations);

    // tensor-scalar inplace invocation
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor_a_arg,
        float scalar,
        BinaryOpType binary_op_type,
        const std::optional<const DataType>& output_dtype,
        tt::stl::Span<const unary::UnaryOpType> lhs_activations,
        tt::stl::Span<const unary::UnaryOpType> rhs_activations,
        tt::stl::Span<const unary::UnaryOpType> post_activations);
};

}  // namespace ttnn::operations::binary_ng

namespace ttnn::prim {
constexpr auto binary_ng =
    ttnn::register_operation<"ttnn::prim::binary_ng", ttnn::operations::binary_ng::BinaryNgDeviceOperation>();
}  // namespace ttnn::prim
