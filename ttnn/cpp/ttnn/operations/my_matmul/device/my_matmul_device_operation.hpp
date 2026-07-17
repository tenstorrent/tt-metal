#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::my_matmul {

struct MyMatmulDeviceOperation {
    // No extra scalar config for this simple op.
    struct operation_attributes_t {};

    // The two matmul operands.
    struct tensor_args_t {
        const Tensor& input_tensor_a;
        const Tensor& input_tensor_b;
    };

    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct SingleCore {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };

    struct MultiCore {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };

    using program_factory_t = std::variant<SingleCore, MultiCore>;  // only one factory

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::my_matmul

namespace ttnn::prim {
ttnn::operations::my_matmul::MyMatmulDeviceOperation::tensor_return_value_t my_matmul(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b);
}  // namespace ttnn::prim
