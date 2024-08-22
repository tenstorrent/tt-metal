// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <variant>

#include "ttnn/common/constants.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/run_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_utils.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::binary {

struct BinaryDeviceOperation {
    struct operation_attributes_t {
        BinaryOpType binary_op_type;
        bool in_place;
        const std::optional<unary::FusedActivations> activations;
        const std::optional<unary::UnaryWithParam> input_tensor_a_activation;
        const MemoryConfig memory_config;
        const DataType dtype;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config;
    };
    struct tensor_args_t {
        const Tensor& input_tensor_a;
        const Tensor& input_tensor_b;
        std::optional<Tensor> output_tensor;
    };
    using shape_return_value_t = ttnn::Shape;
    using tensor_return_value_t = Tensor;

    struct ElementWiseMultiCore {
        struct shared_variables_t {
            KernelHandle binary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
            KernelHandle eltwise_binary_kernel_id;
            CBHandle cb_src0;
            CBHandle cb_src1;
            CBHandle cb_output;
            CoreCoord compute_with_storage_grid_size;
            uint32_t src0_single_tile_size;
            uint32_t src1_single_tile_size;
            uint32_t dst_single_tile_size;
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
            KernelHandle binary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
            KernelHandle bcast_kernel_id;
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
            KernelHandle binary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
            KernelHandle bcast_kernel_id;
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
            KernelHandle binary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
            KernelHandle bcast_kernel_id;
            CoreCoord compute_with_storage_grid_size;
            CBHandle cb_src0;
            uint32_t src0_single_tile_size;
            uint32_t src1_single_tile_size;
            uint32_t dst_single_tile_size;
            CBHandle cb_output;
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
            KernelHandle binary_reader_kernel_id;
            KernelHandle bcast_kernel_id;
            uint32_t cb_src0;
            CBHandle out_cb;
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
            KernelHandle binary_reader_kernel_id;
            KernelHandle bcast_kernel_id;
            uint32_t cb_src0;
            CBHandle out_cb;
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
        BroadcastWidthMultiCore,
        BroadcastHeightMultiCore,
        BroadcastHeightAndWidthMultiCore,
        BroadcastHeightMultiCoreSharded,
        BroadcastHeightMultiCoreShardedOptimized>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static shape_return_value_t compute_output_shapes(
        const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t&, const tensor_args_t&);

    static operation::OpPerformanceModel create_op_performance_model(
        const operation_attributes_t& attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor_a_arg,
        const Tensor& input_tensor_b_arg,
        BinaryOpType binary_op_type,
        bool in_place,
        const std::optional<const DataType>& output_dtype,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> optional_output_tensor,
        std::optional<unary::FusedActivations> activations,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation);
};

}  // namespace ttnn::operations::binary


namespace ttnn::prim {
constexpr auto binary = ttnn::register_operation<"ttnn::prim::binary", ttnn::operations::binary::BinaryDeviceOperation>();
} // namespace ttnn::prim
