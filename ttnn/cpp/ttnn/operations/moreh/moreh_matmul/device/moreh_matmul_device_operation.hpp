// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::moreh::moreh_matmul {
struct MorehMatmulOperation {
    struct operation_attributes_t {
        bool transpose_input;
        bool transpose_other;

        const MemoryConfig output_memory_config;
        const std::optional<DeviceComputeKernelConfig> compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& input;
        const Tensor& other;

        const std::optional<Tensor>& output;
        const std::optional<const Tensor>& bias;
    };

    using shape_return_value_t = Shape;
    using tensor_return_value_t = Tensor;

    struct MultiCoreProgramFactory {
        struct shared_variable_t {
            KernelHandle reader_kernel_id;
            KernelHandle writer_kernel_id;
            std::size_t num_cores;
            std::size_t num_cores_y;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variable_t>;

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

    using program_factory_t = std::variant<MultiCoreProgramFactory>;

    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const Tensor& other,
        bool transpose_input,
        bool transpose_other,
        const std::optional<Tensor>& output,
        const std::optional<const Tensor>& bias,
        const std::optional<MemoryConfig>& output_memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};

void get_tensor_dim(std::vector<uint32_t>& dim, const tt::tt_metal::Shape& shape);
std::vector<int64_t> find_reduce_dim(const tt::tt_metal::Shape& a_shape, const tt::tt_metal::Shape& b_shape);
bool is_same_batch_dim(const Tensor& tensor_a, const Tensor& tensor_b);

}  // namespace ttnn::operations::moreh::moreh_matmul

namespace ttnn::prim {
constexpr auto moreh_matmul =
    ttnn::register_operation<"ttnn::prim::moreh_matmul", ttnn::operations::moreh::moreh_matmul::MorehMatmulOperation>();
}
