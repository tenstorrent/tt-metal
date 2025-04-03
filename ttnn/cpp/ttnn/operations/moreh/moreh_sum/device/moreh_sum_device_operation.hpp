// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <cstddef>
#include <optional>
#include <tuple>
#include <variant>

#include <tt-metalium/kernel_types.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations {
namespace moreh {
namespace moreh_sum {
struct MorehSumOperation::MorehSumHFactory::shared_variables_t;
struct MorehSumOperation::operation_attributes_t;
struct MorehSumOperation::tensor_args_t;
}  // namespace moreh_sum
}  // namespace moreh
}  // namespace operations
}  // namespace ttnn

#define MOREH_SUM_FACTORY_H(name)                                                           \
    struct name {                                                                           \
        struct shared_variables_t {                                                         \
            tt::tt_metal::KernelHandle unary_reader_kernel_id;                              \
            tt::tt_metal::KernelHandle unary_writer_kernel_id;                              \
            std::size_t num_cores;                                                          \
            std::size_t num_cores_y;                                                        \
        };                                                                                  \
                                                                                            \
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>; \
                                                                                            \
        static cached_program_t create(                                                     \
            const operation_attributes_t& operation_attributes,                             \
            const tensor_args_t& tensor_args,                                               \
            tensor_return_value_t& output_tensor);                                          \
                                                                                            \
        static void override_runtime_arguments(                                             \
            cached_program_t& cached_program,                                               \
            const operation_attributes_t& operation_attributes,                             \
            const tensor_args_t& tensor_args,                                               \
            tensor_return_value_t& output_tensor);                                          \
    };

namespace ttnn::operations::moreh::moreh_sum {
struct MorehSumOperation {
    struct operation_attributes_t {
        const int64_t dim;
        const bool keepdim;

        const MemoryConfig memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& input;
        const std::optional<Tensor>& output;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    MOREH_SUM_FACTORY_H(MorehSumHFactory)
    MOREH_SUM_FACTORY_H(MorehSumNCFactory)
    MOREH_SUM_FACTORY_H(MorehSumWFactory)
    MOREH_SUM_FACTORY_H(MorehSumHIntFactory)
    MOREH_SUM_FACTORY_H(MorehSumNCIntFactory)
    MOREH_SUM_FACTORY_H(MorehSumWIntFactory)

    using program_factory_t = std::variant<
        MorehSumHFactory,
        MorehSumNCFactory,
        MorehSumWFactory,
        MorehSumHIntFactory,
        MorehSumNCIntFactory,
        MorehSumWIntFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const int64_t dim,
        const bool keepdim,
        const std::optional<Tensor>& output,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};

}  // namespace ttnn::operations::moreh::moreh_sum

namespace ttnn::prim {
constexpr auto moreh_sum =
    ttnn::register_operation<"ttnn::prim::moreh_sum", ttnn::operations::moreh::moreh_sum::MorehSumOperation>();
}  // namespace ttnn::prim
