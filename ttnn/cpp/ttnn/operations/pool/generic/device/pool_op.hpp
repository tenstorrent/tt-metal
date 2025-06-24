// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations {
namespace pool {
// Generic pool uop -- called from the macro-ops
struct Pool2D {
    struct operation_attributes_t {
        sliding_window::SlidingWindowConfig sliding_window_config_;
        Pool2DType pool_type_;
        DataType output_dtype_;
        MemoryConfig memory_config_;
        bool count_include_pad_;
        std::optional<int32_t> divisor_override_;
        uint32_t memory_used;
    };

    struct tensor_args_t {
        const Tensor& input_tensor_;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct MultiCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader0_kernel;
            tt::tt_metal::KernelHandle reader1_kernel;
            tt::tt_metal::CBHandle raw_in_cb;
            tt::tt_metal::CBHandle cb_out;
            uint32_t ncores;
            uint32_t ncores_w;
            tt::tt_metal::DeviceStorage reader_indices_storage;
            tt::tt_metal::DeviceStorage scalar_config_storage;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);
        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);
    };

    using program_factory_t = std::variant<MultiCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static Tensor create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const Tensor&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const sliding_window::SlidingWindowConfig& sliding_window_config,
        Pool2DType pool_type,
        DataType output_dtype,
        MemoryConfig memory_config,
        bool count_include_pad,
        std::optional<int32_t> divisor_override,
        uint32_t memory_used);
};

}  // namespace pool
}  // namespace ttnn::operations

namespace ttnn::prim {
constexpr auto pool2d = ttnn::register_operation<"ttnn::prim::pool2d", ttnn::operations::pool::Pool2D>();
}  // namespace ttnn::prim
