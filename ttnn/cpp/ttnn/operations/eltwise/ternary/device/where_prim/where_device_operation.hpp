// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <type_traits>
#include <variant>

#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/command_queue.hpp>

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/ternary/common/ternary_op_types.hpp"

namespace ttnn::operations::ternary {

// TODO: Name collision with variant
template <typename T>
concept FloatOrTensorConcept = std::is_same_v<T, Tensor> || std::floating_point<T>;

struct WhereDeviceOperation {
    using tensor_return_value_t = Tensor;
    using spec_return_value_t = TensorSpec;

    struct operation_attributes_t {
        TernaryOpType ternary_op_type;
        const std::optional<float> b_scalar;
        const std::optional<float> c_scalar;
        const MemoryConfig memory_config;
        const DataType dtype;
        const CoreRangeSet worker_grid;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config;

        tt::stl::hash::hash_t to_hash() const {
            // hash has to exclude the scalar value
            return tt::stl::hash::hash_objects_with_default_seed(
                ternary_op_type, memory_config, dtype, compute_kernel_config);
        }
    };
    struct tensor_args_t {
        const Tensor& input_tensor_a;
        std::optional<Tensor> input_tensor_b;
        std::optional<Tensor> input_tensor_c;
        std::optional<Tensor> output_tensor;
    };

    struct ElementWiseMultiCoreWhereProgram {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
            tt::tt_metal::KernelHandle eltwise_kernel_id;
            tt::tt_metal::CBHandle cb_src0;
            tt::tt_metal::CBHandle cb_src1;
            tt::tt_metal::CBHandle cb_src2;
            tt::tt_metal::CBHandle cb_output;
            CoreRangeSet all_device_cores;
            uint32_t src0_single_tile_size;
            uint32_t src1_single_tile_size;
            uint32_t src2_single_tile_size;
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

    struct BroadcastScalarsWhereProgram {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
            tt::tt_metal::KernelHandle broadcast_kernel_id;
            tt::tt_metal::CBHandle cb_src0;
            tt::tt_metal::CBHandle cb_output;
            CoreRangeSet all_device_cores;
            uint32_t src0_single_tile_size;
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

    using program_factory_t = std::variant<ElementWiseMultiCoreWhereProgram, BroadcastScalarsWhereProgram>;
    // ElementWiseTensorsWhereProgram
    // BroadcastTensorScalarWhereProgram
    //
    // +sfpu
    // +Sharded

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModel create_op_performance_model(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args, Tensor& tensor_return_value);

    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const Tensor&);

    template <FloatOrTensorConcept BType, FloatOrTensorConcept CType>
    static CoreRangeSet initWorkerGrid(
        const Tensor& cond_tensor,
        const BType& input_true,
        const CType& input_false,
        std::optional<Tensor>& output_tensor) {
        // We assert all shard specs are the same if sharded, so only need to check the first shard spec
        // This will create the worker grid based on the used sub-devices when sharded
        // Otherwise this will use all cores of the sub-devices
        // TODO #13655: Note that the current program ingfrastructure still only supports a single sub-device per
        // program

        auto initWorkerGridImpl = [](const Tensor& tensor) {
            CoreRangeSet worker_grid;
            const auto& input_grid = tensor.shard_spec().value().grid;
            auto device = tensor.device();
            for (const auto& sub_device_id : device->get_sub_device_ids()) {
                const auto& sub_device_workers =
                    device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id);
                if (sub_device_workers.intersects(input_grid)) {
                    worker_grid = worker_grid.merge(sub_device_workers);
                }
            }
            return worker_grid;
        };

        CoreRangeSet worker_grid;
        if (cond_tensor.is_sharded()) {
            return initWorkerGridImpl(cond_tensor);
        }
        if constexpr (std::is_same_v<BType, Tensor>) {
            if (input_true.is_sharded()) {
                return initWorkerGridImpl(input_true);
            }
        }
        if constexpr (std::is_same_v<CType, Tensor>) {
            if (input_false.is_sharded()) {
                return initWorkerGridImpl(input_false);
            }
        }
        if (output_tensor.has_value() && output_tensor->is_sharded()) {
            return initWorkerGridImpl(*output_tensor);
        }

        auto device = cond_tensor.device();
        for (const auto& sub_device_id : device->get_sub_device_ids()) {
            const auto& sub_device_workers =
                device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id);
            worker_grid = worker_grid.merge(sub_device_workers);
        }

        return worker_grid;
    }

    // can't use invoke template, we check existance of the invoke function before template instantiation
    template <FloatOrTensorConcept BType, FloatOrTensorConcept CType>
    static std::tuple<operation_attributes_t, tensor_args_t> invoke_impl(
        TernaryOpType ternary_op_type,
        const Tensor& a_tensor,
        const BType& b_tensor,
        const CType& c_tensor,
        const std::optional<const DataType>& dtype,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> output_tensor) {
        // TODO: Should we do that check earlier?
        if (dtype.has_value() && output_tensor.has_value()) {
            TT_FATAL(
                dtype.value() == output_tensor.value().get_dtype(),
                "If both output dtype and output tensor provided dtype should match");
        }

        auto worker_grid = initWorkerGrid(a_tensor, b_tensor, c_tensor, output_tensor);

        // TODO: Improve implementation
        constexpr auto fetchScalar = [](auto&& scalar) {
            if constexpr (std::is_floating_point_v<std::decay_t<decltype(scalar)>>) {
                return scalar;
            } else {
                return std::nullopt;
            }
        };

        constexpr auto fetchTensor = [](auto&& tensor) {
            if constexpr (std::is_same_v<Tensor, std::decay_t<decltype(tensor)>>) {
                return tensor;
            } else {
                return std::nullopt;
            }
        };

        return {
            operation_attributes_t{
                .ternary_op_type = ternary_op_type,
                .b_scalar = fetchScalar(b_tensor),
                .c_scalar = fetchScalar(c_tensor),
                .memory_config = memory_config.value_or(
                    output_tensor.has_value() ? output_tensor->memory_config() : a_tensor.memory_config()),
                .dtype = dtype.value_or(a_tensor.get_dtype()),
                .worker_grid = std::move(worker_grid),
                .compute_kernel_config = std::nullopt},
            tensor_args_t{
                .input_tensor_a = a_tensor,
                .input_tensor_b = fetchTensor(b_tensor),
                .input_tensor_c = fetchTensor(c_tensor),
                .output_tensor = output_tensor}};
    }

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        TernaryOpType ternary_op_type,
        const Tensor& a_tensor,
        const Tensor& b_tensor,
        const Tensor& c_tensor,
        const std::optional<const DataType>& dtype,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> output_tensor) {
        return invoke_impl(
            ternary_op_type, a_tensor, b_tensor, c_tensor, dtype, memory_config, std::move(output_tensor));
    }

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        TernaryOpType ternary_op_type,
        const Tensor& a_tensor,
        float b_tensor,
        const Tensor& c_tensor,
        const std::optional<const DataType>& dtype,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> output_tensor) {
        return invoke_impl(
            ternary_op_type, a_tensor, b_tensor, c_tensor, dtype, memory_config, std::move(output_tensor));
    }

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        TernaryOpType ternary_op_type,
        const Tensor& a_tensor,
        const Tensor& b_tensor,
        float c_tensor,
        const std::optional<const DataType>& dtype,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> output_tensor) {
        return invoke_impl(
            ternary_op_type, a_tensor, b_tensor, c_tensor, dtype, memory_config, std::move(output_tensor));
    }

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        TernaryOpType ternary_op_type,
        const Tensor& a_tensor,
        float b_tensor,
        float c_tensor,
        const std::optional<const DataType>& dtype,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> output_tensor) {
        return invoke_impl(
            ternary_op_type, a_tensor, b_tensor, c_tensor, dtype, memory_config, std::move(output_tensor));
    }
};

static_assert(
    ttnn::decorators::PrimitiveOperationConcept<WhereDeviceOperation>,
    "WhereDeviceOperation must satisfy PrimitiveOperationConcept ");

}  // namespace ttnn::operations::ternary

namespace ttnn::prim {
// TODO: WhereDeviceOperation could be renamed to TernaryDeviceOperation
constexpr auto where_impl =
    ttnn::register_operation<"ttnn::prim::where_impl", ttnn::operations::ternary::WhereDeviceOperation>();
}  // namespace ttnn::prim
