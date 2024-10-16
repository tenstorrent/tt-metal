// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#define DEFINE_PROGRAM_FACTORY(FactoryName)                                                        \
    struct FactoryName {                                                                           \
        struct shared_variables_t {                                                                \
            KernelHandle reader_kernels_id;                                                        \
            KernelHandle writer_kernels_id;                                                        \
            std::size_t num_cores_to_be_used;                                                      \
            std::size_t num_cores_y;                                                               \
        };                                                                                         \
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;        \
        static cached_program_t create(const operation_attributes_t& operation_attributes,         \
                                       const tensor_args_t& tensor_args,                           \
                                       tensor_return_value_t& output);                             \
        static void override_runtime_arguments(cached_program_t& cached_program,                   \
                                               const operation_attributes_t& operation_attributes, \
                                               const tensor_args_t& tensor_args,                   \
                                               tensor_return_value_t& output);                     \
    };

namespace ttnn::operations::moreh::moreh_norm_backward {

std::tuple<uint32_t, float, bool> get_floored_p_and_decimal_and_p_is_negative(float p);
void get_tensor_dim(std::vector<uint32_t>& dim, const Shape& shape);
tt::tt_metal::LegacyShape get_output_grad_shape(const Tensor& output_grad,
                                                const Tensor& input_grad,
                                                const std::vector<int64_t>& dims,
                                                const bool& keepdim);

struct MorehNormBackwardOperation {
    struct operation_attributes_t {
        float p;
        std::vector<int64_t> dims;
        bool keepdim;
        const MemoryConfig memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& input;
        const Tensor& output;
        const Tensor& output_grad;
        const std::optional<Tensor>& input_grad;
    };

    using shape_return_value_t = Shape;
    using tensor_return_value_t = Tensor;

    DEFINE_PROGRAM_FACTORY(ProgramFactory)

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const Tensor& output,
        const Tensor& output_grad,
        float p,
        std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
        bool keepdim,
        const std::optional<Tensor>& input_grad,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};

}  // namespace ttnn::operations::moreh::moreh_norm_backward

namespace ttnn::prim {
constexpr auto moreh_norm_backward =
    ttnn::register_operation<"ttnn::prim::moreh_norm_backward",
                             ttnn::operations::moreh::moreh_norm_backward::MorehNormBackwardOperation>();
}
