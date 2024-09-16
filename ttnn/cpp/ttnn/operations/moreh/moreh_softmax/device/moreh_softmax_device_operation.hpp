// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"


namespace ttnn::operations::moreh::moreh_softmax {

enum class MorehSoftmaxOpParallelizationStrategy {
    NONE,
    SMALL_W,
    SMALL_H,
    LARGE_W,
    LARGE_H,
    LARGE_C,
};

enum class MorehSoftmaxOp {
    SOFTMAX,
    SOFTMIN,
    LOGSOFTMAX,
};

bool is_moreh_softmax_w_small_available(const Tensor &tensor, std::optional<DeviceComputeKernelConfig> compute_kernel_config);
bool is_moreh_softmax_h_small_available(const Tensor &tensor, std::optional<DeviceComputeKernelConfig> compute_kernel_config);

struct MorehSoftmaxOperation {
    struct operation_attributes_t {
        const uint32_t dim;
        const MorehSoftmaxOp op;
        const MorehSoftmaxOpParallelizationStrategy strategy;
        const MemoryConfig output_memory_config;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor &input_tensor;
        const std::optional<Tensor> &output_tensor;
    };

    using shape_return_value_t = ttnn::Shape;
    using tensor_return_value_t = Tensor;

    struct MorehSoftmaxCLargeFactory {
        struct shared_variables_t {
            KernelHandle unary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
            std::size_t num_cores;
            std::size_t num_cores_y;
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

    struct MorehSoftmaxHLargeFactory {
        struct shared_variables_t {
            KernelHandle unary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
            std::size_t num_cores;
            std::size_t num_cores_y;
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

    struct MorehSoftmaxHSmallFactory {
        struct shared_variables_t {
            KernelHandle unary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
            std::size_t num_cores;
            std::size_t num_cores_y;
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

    struct MorehSoftmaxWLargeFactory {
        struct shared_variables_t {
            KernelHandle unary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
            std::size_t num_cores;
            std::size_t num_cores_y;
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

    struct MorehSoftmaxWSmallFactory {
        struct shared_variables_t {
            KernelHandle unary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
            std::size_t num_cores;
            std::size_t num_cores_y;
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
        MorehSoftmaxCLargeFactory,
        MorehSoftmaxHLargeFactory,
        MorehSoftmaxWLargeFactory,
        MorehSoftmaxHSmallFactory,
        MorehSoftmaxWSmallFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_with_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static MorehSoftmaxOpParallelizationStrategy get_parallelization_strategy(
        const Tensor &input_tensor,
        const std::optional<Tensor> &output_tensor,
        const uint32_t dim,
        const MorehSoftmaxOpParallelizationStrategy strategy,
        const std::optional<MemoryConfig> output_memory_config,
        const std::optional<DeviceComputeKernelConfig> compute_kernel_config);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor &input_tensor,
        const uint32_t dim,
        const std::optional<Tensor> &output_tensor,
        const MorehSoftmaxOp op,
        const MorehSoftmaxOpParallelizationStrategy strategy,
        const std::optional<MemoryConfig> output_memory_config,
        const std::optional<DeviceComputeKernelConfig> compute_kernel_config);
    };

}

namespace ttnn::prim {
constexpr auto moreh_softmax =
    ttnn::register_operation<"ttnn::prim::moreh_softmax", ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOperation>();
}
