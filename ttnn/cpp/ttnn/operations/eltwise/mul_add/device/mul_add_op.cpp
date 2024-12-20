#include "mul_add_op.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::mul_add {

MulAddDeviceOperation::program_factory_t MulAddDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t&) {
    return MulAddProgramFactorySingleCore();
}

void MulAddDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {}
void MulAddDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {}

TensorSpec MulAddDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return {tensor_args.input_tensor_a.get_tensor_spec()};
}

Tensor MulAddDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return {create_device_tensor(tensor_args.input_tensor_a.get_tensor_spec(), tensor_args.input_tensor_a.device())};
}

std::tuple<MulAddDeviceOperation::operation_attributes_t, MulAddDeviceOperation::tensor_args_t>
MulAddDeviceOperation::invoke(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, const Tensor& input_tensor_c) {
    return {operation_attributes_t{true}, tensor_args_t{input_tensor_a, input_tensor_b, input_tensor_c}};
}

}  // namespace ttnn::operations::mul_add
