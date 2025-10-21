// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/ternary/common/ternary_op_types.hpp"

namespace ttnn::operations::ternary {

struct TernaryDeviceOperation {
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct operation_attributes_t {
        TernaryOpType ternary_op_type;
        TernaryVariant ternary_variant;
        TernaryBroadcastType broadcast_type;
        tt::tt_metal::MemoryConfig memory_config;
        DataType input_dtype;
        std::optional<DataType> dtype;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config;

        // Scalar values for TTS/TST/TSS variants
        std::optional<float> scalar_input_a;  // For TST/TSS
        std::optional<float> scalar_input_b;  // For TTS/TSS

        tt::stl::hash::hash_t to_hash() const;

        DataType get_dtype() const;
    };

    struct tensor_args_t {
        const Tensor& input_tensor_a;          // First input (predicate for where, input for lerp)
        std::optional<Tensor> input_tensor_b;  // Second input (value_true for where, end for lerp)
        std::optional<Tensor> input_tensor_c;  // Third input (value_false for where, weight for lerp)
        std::optional<Tensor> optional_output_tensor;
    };

    struct TernaryProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id{};
            tt::tt_metal::KernelHandle writer_kernel_id{};
            tt::tt_metal::KernelHandle compute_kernel_id{};
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

    using program_factory_t = std::variant<TernaryProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);

    // tensor-tensor-tensor invocation (TTT)
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        TernaryOpType op_type,
        const Tensor& input_a,
        const Tensor& input_b,
        const Tensor& input_c,
        const std::optional<const DataType>& output_dtype,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<Tensor>& optional_output_tensor);

    // tensor-tensor-scalar invocation (TTS)
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        TernaryOpType op_type,
        const Tensor& input_a,
        const Tensor& input_b,
        float scalar_c,
        const std::optional<const DataType>& output_dtype,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<Tensor>& optional_output_tensor);

    // tensor-scalar-tensor invocation (TST)
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        TernaryOpType op_type,
        const Tensor& input_a,
        float scalar_b,
        const Tensor& input_c,
        const std::optional<const DataType>& output_dtype,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<Tensor>& optional_output_tensor);
};

}  // namespace ttnn::operations::ternary

namespace ttnn::prim {
constexpr auto ternary =
    ttnn::register_operation<"ttnn::prim::ternary", ttnn::operations::ternary::TernaryDeviceOperation>();
}  // namespace ttnn::prim
