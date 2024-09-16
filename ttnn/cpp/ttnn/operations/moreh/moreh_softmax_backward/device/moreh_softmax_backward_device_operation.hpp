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


namespace ttnn::operations::moreh::moreh_softmax_backward {

enum class MorehSoftmaxBackwardOpParallelizationStrategy {
    NONE,
    SMALL_W,
    SMALL_H,
    LARGE_W,
    LARGE_H,
    LARGE_C,
};

enum class MorehSoftmaxBackwardOp {
    SOFTMAX,
    SOFTMIN,
    LOGSOFTMAX,
};

bool is_moreh_softmax_backward_w_small_available(const Tensor &tensor);
bool is_moreh_softmax_backward_h_small_available(const Tensor &tensor);
struct MorehSoftmaxBackwardOperation {
    struct operation_attributes_t {
        const uint32_t dim;
        const MorehSoftmaxBackwardOp op;
        const MorehSoftmaxBackwardOpParallelizationStrategy strategy;
        const MemoryConfig output_memory_config;
        const std::optional<DeviceComputeKernelConfig> compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor &output_tensor;
        const Tensor &output_grad_tensor;
        const std::optional<Tensor> &input_grad_tensor;
    };

    using shape_return_value_t = ttnn::Shape;
    using tensor_return_value_t = Tensor;

    struct MorehSoftmaxBackwardCLargeFactory {
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

    struct MorehSoftmaxBackwardHLargeFactory {
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

    struct MorehSoftmaxBackwardHSmallFactory {
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

    struct MorehSoftmaxBackwardWLargeFactory {
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

    struct MorehSoftmaxBackwardWSmallFactory {
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
        MorehSoftmaxBackwardCLargeFactory,
        MorehSoftmaxBackwardHLargeFactory,
        MorehSoftmaxBackwardWLargeFactory,
        MorehSoftmaxBackwardHSmallFactory,
        MorehSoftmaxBackwardWSmallFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_with_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor &output_tensor,
        const Tensor &output_grad_tensor,
        const uint32_t dim,
        const std::optional<Tensor> &input_grad_tensor,
        const MorehSoftmaxBackwardOp op,
        const MorehSoftmaxBackwardOpParallelizationStrategy strategy,
        const std::optional<MemoryConfig> output_memory_config,
        const std::optional<DeviceComputeKernelConfig> compute_kernel_config);
    static MorehSoftmaxBackwardOpParallelizationStrategy get_parallelization_strategy(
        const Tensor &output_tensor,
        const Tensor &output_grad_tensor,
        const uint32_t dim,
        const std::optional<Tensor> &input_grad_tensor,
        const MorehSoftmaxBackwardOp op,
        const MorehSoftmaxBackwardOpParallelizationStrategy strategy,
        const std::optional<MemoryConfig> output_memory_config,
        const std::optional<DeviceComputeKernelConfig> compute_kernel_config);
    };
}

namespace ttnn::prim {
constexpr auto moreh_softmax_backward =
    ttnn::register_operation<"ttnn::prim::moreh_softmax_backward", ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOperation>();
}
