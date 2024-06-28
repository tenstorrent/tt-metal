// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <variant>

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
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::binary {

constexpr uint8_t DefaultQueueId = 0;
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

std::map<string, string> get_defines(
    BinaryOpType op_type,
    const std::optional<DataType> in_dtype = std::nullopt,
    const std::optional<DataType> out_dtype = std::nullopt,
    const std::optional<FusedActivations> fused_activations = std::nullopt);

}  // namespace utils

struct BinaryDeviceOperation {
    struct operation_attributes_t {
        BinaryOpType binary_op_type;
        bool in_place;
        const std::optional<FusedActivations> activations;
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

    using program_factory_t = std::variant<
        ElementWiseMultiCore,
        BroadcastWidthMultiCore,
        BroadcastHeightMultiCore,
        BroadcastHeightAndWidthMultiCore>;

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
};

}  // namespace ttnn::operations::binary
