// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
// #include <variant>

#include "ttnn/tensor/tensor.hpp"
// #include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
// #include "ttnn/types.hpp"
// #include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::reduction {

constexpr int DIM_DEFAULT_VALUE = -1;
constexpr bool DESCENDING_DEFAULT_VALUE = false;
constexpr bool STABLE_DEFAULT_VALUE = -1;
// constexpr std::optional<std::tuple<Tensor&, Tensor&>> OPTIONAL_INPUT_OUTPUT_TENSORS_DEFAULT_VALUE = std::nullopt;

struct SortDeviceOperation {
    // struct operation_attributes_t {
    const int8_t dim;
    const bool descending;
    const bool stable;
    const tt::tt_metal::MemoryConfig output_mem_config;
    // };

    // struct tensor_args_t {
    //     const Tensor& input_tensor;

    //     // If optional input/optput tensors are provided the output_tensors will point to the memory of the provided
    //     // tensors
    //     std::optional<std::tuple<Tensor&, Tensor&>> optional_input_output_tensors;

    //     std::tuple<Tensor, Tensor> output_tensors;
    // };

    // using spec_return_value_t = std::tuple<ttnn::TensorSpec, ttnn::TensorSpec>;
    // using tensor_return_value_t = std::variant<std::tuple<Tensor, Tensor>, std::tuple<Tensor&, Tensor&>>;

    void validate_with_output_tensors(
        const Tensor& input_tensor, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<TensorSpec> compute_output_specs(
        const Tensor& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const Tensor& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const Tensor& input_tensors, std::vector<Tensor>& output_tensors) const;
    // static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    // static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    // static std::tuple<operation_attributes_t, tensor_args_t> invoke(
    //     const Tensor& input_tensor,
    //     std::tuple<Tensor, Tensor> output_tensors,
    //     const int dim = DIM_DEFAULT_VALUE,
    //     const bool descending = DESCENDING_DEFAULT_VALUE,
    //     const bool stable = STABLE_DEFAULT_VALUE,
    //     std::optional<std::tuple<Tensor&, Tensor&>> optional_input_output_tensors =
    //         OPTIONAL_INPUT_OUTPUT_TENSORS_DEFAULT_VALUE);
};

}  // namespace ttnn::operations::experimental::reduction
