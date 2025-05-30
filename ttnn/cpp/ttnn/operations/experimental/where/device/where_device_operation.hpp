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

namespace ttnn::operations::experimental::where {

template <typename T>
concept FloatOrTensorConcept = std::is_same_v<T, Tensor> || std::floating_point<T>;

struct WhereDeviceOperation {
    using tensor_return_value_t = Tensor;
    using spec_return_value_t = TensorSpec;

    struct operation_attributes_t {
        const MemoryConfig memory_config;
        const DataType dtype;
        const CoreRangeSet worker_grid;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config;

        tt::stl::hash::hash_t to_hash() const {
            // hash has to exclude the scalar value
            return tt::stl::hash::hash_objects_with_default_seed(memory_config, dtype, compute_kernel_config);
        }
    };
    struct tensor_args_t {
        const Tensor& input_tensor_a;
        Tensor input_tensor_b;
        Tensor input_tensor_c;
        std::optional<Tensor> output_tensor;
    };

    // move to cpp files
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

    using program_factory_t = std::variant<ElementWiseMultiCoreWhereProgram>;

    // PrimitiveOperationConcept methods
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModel create_op_performance_model(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args, Tensor& tensor_return_value);

    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const Tensor&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& a_tensor,
        const Tensor& b_tensor,
        const Tensor& c_tensor,
        const std::optional<const DataType>& dtype,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> output_tensor);
};

static_assert(
    ttnn::decorators::PrimitiveOperationConcept<WhereDeviceOperation>,
    "WhereDeviceOperation must satisfy PrimitiveOperationConcept");

}  // namespace ttnn::operations::experimental::where

namespace ttnn::prim {

constexpr auto where_impl =
    ttnn::register_operation<"ttnn::prim::where_impl", ttnn::operations::experimental::where::WhereDeviceOperation>();

}  // namespace ttnn::prim
