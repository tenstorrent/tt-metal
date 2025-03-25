// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sort_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::reduction {

// SortDeviceOperation::program_factory_t SortDeviceOperation::select_program_factory(
//     const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
//     bool some_condition_based_on_operation_attributes_and_or_tensor_args = true;
//     if (some_condition_based_on_operation_attributes_and_or_tensor_args) {
//         return SingleCore{};
//     }
//     return MultiCore{};
// }

// void SortDeviceOperation::validate_on_program_cache_miss(
//     const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

// void SortDeviceOperation::validate_on_program_cache_hit(
//     const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

// SortDeviceOperation::spec_return_value_t SortDeviceOperation::compute_output_specs(
//     const operation_attributes_t&, const tensor_args_t& tensor_args) {
//     const auto& input_tensor = tensor_args.input_tensor;
//     TensorSpec spec(
//         input_tensor.get_logical_shape(),
//         tt::tt_metal::TensorLayout(
//             input_tensor.get_dtype(), tt::tt_metal::PageConfig(input_tensor.get_layout()), MemoryConfig{}));
//     return {spec, spec};
// }

// SortDeviceOperation::tensor_return_value_t SortDeviceOperation::create_output_tensors(
//     const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
//     auto output_spec = compute_output_specs(operation_attributes, tensor_args);

//     // If optional input/output tensors are provided
//     if (tensor_args.optional_input_output_tensors.has_value()) {
//         // TODO: Add checking if tensors have apropriate shapes
//         return tensor_args.optional_input_output_tensors.value();
//     }

//     // Create output tensors
//     Tensor output_tensor =
//         create_device_tensor(std::get<0>(output_spec), std::get<0>(tensor_args.output_tensors).device());
//     Tensor output_indices_tensor =
//         create_device_tensor(std::get<1>(output_spec), std::get<1>(tensor_args.output_tensors).device());

//     return std::tie(output_tensor, output_indices_tensor);
// }

// static std::tuple<SortDeviceOperation::operation_attributes_t, SortDeviceOperation::tensor_args_t> invoke(
//     const Tensor& input_tensor,
//     std::tuple<Tensor, Tensor> output_tensors,
//     const int dim,
//     const bool descending,
//     const bool stable,
//     std::optional<std::tuple<Tensor&, Tensor&>> optional_input_output_tensors) {
//     return {
//         SortDeviceOperation::operation_attributes_t{.dim = dim, .descending = descending, .stable = stable},
//         SortDeviceOperation::tensor_args_t{
//             .input_tensor = input_tensor,
//             .optional_input_output_tensors = optional_input_output_tensors,
//             .output_tensors = std::move(output_tensors)}};
// }

void SortDeviceOperation::validate_with_output_tensors(
    const Tensor& input_tensor, const std::vector<std::optional<Tensor>>& output_tensors) const {
    // TODO: DEBUG:
    tt::log_error("1. validate_with_output_tensors !!!!!!!!!!!!");
    // ---
    // Validate shapes of input and output tensors
    const auto input_tensor_shape = input_tensor.get_padded_shape();
    if (output_tensors.size() == 2) {
        if (output_tensors.at(0).has_value() && output_tensors.at(1).has_value()) {
            const auto output_tensor_shape = output_tensors.at(0)->get_padded_shape();
            TT_FATAL(
                output_tensor_shape != input_tensor_shape,
                "Output tensor shape must be the same as input tensor shape");

            const auto output_indices_shape = output_tensors.at(1)->get_padded_shape();
            TT_FATAL(
                output_indices_shape != input_tensor_shape,
                "Output tensor indices shape must be the same as input tensor shape");
        }
    }
}

std::vector<TensorSpec> SortDeviceOperation::compute_output_specs(
    const Tensor& input_tensor, const std::vector<std::optional<Tensor>>& output_tensors) const {
    // TODO: DEBUG:
    tt::log_error("2. compute_output_specs !!!!!!!!!!!!");
    // std::cout << "2. compute_output_specs !!!!!!!!!!!!" << std::endl;
    // ---
    if (output_tensors.size() == 2) {
        if (output_tensors.at(0).has_value() && output_tensors.at(1).has_value()) {
            return {output_tensors[0]->get_tensor_spec(), output_tensors[1]->get_tensor_spec()};
        }
    }
    // Create output tensors specs
    auto output_shape = input_tensor.get_logical_shape();
    auto values_spec =
        TensorSpec(output_shape, TensorLayout(input_tensor.get_dtype(), PageConfig(Layout::TILE), output_mem_config));
    auto index_spec =
        TensorSpec(output_shape, TensorLayout(DataType::UINT16, PageConfig(Layout::TILE), output_mem_config));

    return {values_spec, index_spec};
}

std::vector<Tensor> SortDeviceOperation::create_output_tensors(
    const Tensor& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    // TODO: DEBUG:
    tt::log_error("3. create_output_tensors !!!!!!!!!!!!");
    // std::cout << "3. create_output_tensors !!!!!!!!!!!!" << std::endl;
    // ---
    if (output_tensors.size() == 2) {
        if (output_tensors.at(0).has_value() && output_tensors.at(1).has_value()) {
            return {output_tensors[0].value(), output_tensors[1].value()};
        }
    }
    auto output_specs = compute_output_specs(input_tensors, output_tensors);
    return {
        create_device_tensor(output_specs[0], input_tensors.device()),
        create_device_tensor(output_specs[1], input_tensors.device()),
    };
}

tt::tt_metal::operation::ProgramWithCallbacks SortDeviceOperation::create_program(
    const Tensor& input_tensors, std::vector<Tensor>& output_tensors) const {
    // TODO: DEBUG:
    tt::log_error("4. create_program !!!!!!!!!!!!");
    // std::cout << "4. create_program !!!!!!!!!!!!" << std::endl;
    // ---
    return {};  // TODO: Change this
}
}  // namespace ttnn::operations::experimental::reduction
