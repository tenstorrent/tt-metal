// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <variant>

#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::binary {

struct BinaryDeviceOperation {
    struct operation_attributes_t {
        BinaryOpType binary_op_type{};
        const std::optional<unary::EltwiseFusedActivations> activations;
        const std::optional<unary::EltwiseUnaryWithParam> input_tensor_a_activation;
        const std::optional<float> scalar;
        const MemoryConfig memory_config;
        const DataType dtype{};
        const CoreRangeSet worker_grid;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config;

        tt::stl::hash::hash_t to_hash() const {
            // hash has to exclude the scalar value
            return tt::stl::hash::hash_objects_with_default_seed(
                binary_op_type, activations, input_tensor_a_activation, memory_config, dtype, compute_kernel_config);
        }
    };
    struct tensor_args_t {
        const Tensor& input_tensor_a;
        std::optional<Tensor> input_tensor_b;
        std::optional<Tensor> output_tensor;
    };
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ElementWiseMultiCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle binary_reader_kernel_id{};
            tt::tt_metal::KernelHandle unary_writer_kernel_id{};
            tt::tt_metal::KernelHandle eltwise_binary_kernel_id{};
            tt::tt_metal::CBHandle cb_src0{};
            tt::tt_metal::CBHandle cb_src1{};
            tt::tt_metal::CBHandle cb_output{};
            CoreRangeSet all_device_cores;
            uint32_t src0_single_tile_size{};
            uint32_t src1_single_tile_size{};
            uint32_t dst_single_tile_size{};
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    struct ElementWiseMultiCoreSfpu {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle binary_reader_kernel_id{};
            tt::tt_metal::KernelHandle unary_writer_kernel_id{};
            tt::tt_metal::KernelHandle eltwise_binary_kernel_id{};
            tt::tt_metal::CBHandle cb_src0{};
            tt::tt_metal::CBHandle cb_src1{};
            tt::tt_metal::CBHandle cb_output{};
            CoreRangeSet all_device_cores;
            uint32_t src0_single_tile_size{};
            uint32_t src1_single_tile_size{};
            uint32_t dst_single_tile_size{};
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };
    struct BroadcastWidthMultiCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle binary_reader_kernel_id{};
            tt::tt_metal::KernelHandle unary_writer_kernel_id{};
            tt::tt_metal::KernelHandle bcast_kernel_id{};
            CoreCoord compute_with_storage_grid_size;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    struct BroadcastHeightMultiCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle binary_reader_kernel_id{};
            tt::tt_metal::KernelHandle unary_writer_kernel_id{};
            tt::tt_metal::KernelHandle bcast_kernel_id{};
            CoreCoord compute_with_storage_grid_size;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    struct BroadcastHeightAndWidthMultiCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle binary_reader_kernel_id{};
            tt::tt_metal::KernelHandle unary_writer_kernel_id{};
            tt::tt_metal::KernelHandle bcast_kernel_id{};
            CoreCoord compute_with_storage_grid_size;
            tt::tt_metal::CBHandle cb_src0{};
            uint32_t src0_single_tile_size{};
            uint32_t src1_single_tile_size{};
            uint32_t dst_single_tile_size{};
            tt::tt_metal::CBHandle cb_output{};
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    struct BroadcastHeightMultiCoreSharded {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle binary_reader_kernel_id;
            tt::tt_metal::KernelHandle bcast_kernel_id;
            tt::tt_metal::CBHandle cb_src0;
            tt::tt_metal::CBHandle out_cb;
            uint32_t ncores_x;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    struct BroadcastHeightMultiCoreShardedOptimized {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle binary_reader_kernel_id;
            tt::tt_metal::KernelHandle bcast_kernel_id;
            tt::tt_metal::CBHandle cb_src0;
            tt::tt_metal::CBHandle out_cb;
            uint32_t ncores_x;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<
        ElementWiseMultiCore,
        ElementWiseMultiCoreSfpu,
        BroadcastWidthMultiCore,
        BroadcastHeightMultiCore,
        BroadcastHeightAndWidthMultiCore,
        BroadcastHeightMultiCoreSharded,
        BroadcastHeightMultiCoreShardedOptimized>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);
};

}  // namespace ttnn::operations::binary

namespace ttnn::prim {

ttnn::operations::binary::BinaryDeviceOperation::tensor_return_value_t binary(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    ttnn::operations::binary::BinaryOpType binary_op_type,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<Tensor> optional_output_tensor = std::nullopt,
    std::optional<ttnn::operations::unary::EltwiseFusedActivations> activations = std::nullopt,
    std::optional<ttnn::operations::unary::EltwiseUnaryWithParam> input_tensor_a_activation = std::nullopt);

ttnn::operations::binary::BinaryDeviceOperation::tensor_return_value_t binary(
    const Tensor& input_tensor_a_arg,
    float scalar,
    ttnn::operations::binary::BinaryOpType binary_op_type,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<Tensor> optional_output_tensor = std::nullopt,
    std::optional<ttnn::operations::unary::EltwiseFusedActivations> activations = std::nullopt,
    std::optional<ttnn::operations::unary::EltwiseUnaryWithParam> input_tensor_a_activation = std::nullopt);

}  // namespace ttnn::prim
