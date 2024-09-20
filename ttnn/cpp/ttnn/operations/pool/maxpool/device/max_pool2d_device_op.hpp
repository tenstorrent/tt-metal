// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/cpp/ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/decorators.hpp"


namespace ttnn::operations {
namespace pool {

inline uint32_t ceil_multiple_of(uint32_t n, uint32_t m) {
    return (uint32_t) std::ceil((float) n / m) * m;
}

// new maxpool uop -- called from the macro-op
struct MaxPool2D {
    struct operation_attributes_t {
        sliding_window::SlidingWindowConfig sliding_window_config_;
        DataType output_dtype_;
        MemoryConfig memory_config_;
    };

    struct tensor_args_t {
        const Tensor& input_tensor_;
    };

    using shape_return_value_t = ttnn::Shape;
    using tensor_return_value_t = Tensor;

    struct MultiCore {
        struct shared_variables_t {
            KernelHandle reader0_kernel;
            KernelHandle reader1_kernel;
            CBHandle raw_in_cb;
            CBHandle cb_out;
            uint32_t ncores;
            uint32_t ncores_w;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(const operation_attributes_t& operation_attributes,
                                       const tensor_args_t& tensor_args,
                                       tensor_return_value_t& output_tensor);
        static void override_runtime_arguments(cached_program_t& cached_program,
                                               const operation_attributes_t& operation_attributes,
                                               const tensor_args_t& tensor_args,
                                               tensor_return_value_t& output_tensor);
    };

    using program_factory_t = std::variant<MultiCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);
    static Tensor create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static operation::OpPerformanceModel create_op_performance_model(const operation_attributes_t&, const tensor_args_t&, const Tensor&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const sliding_window::SlidingWindowConfig& sliding_window_config,
        DataType output_dtype,
        MemoryConfig memory_config);

};

}  // namespace pool
}  // namespace ttnn::operations

namespace ttnn::prim {
constexpr auto max_pool2d = ttnn::register_operation<"ttnn::prim::max_pool2d", ttnn::operations::pool::MaxPool2D>();
}  // namespace ttnn::prim
