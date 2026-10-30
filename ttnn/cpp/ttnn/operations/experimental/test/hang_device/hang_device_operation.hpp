// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct ExecuteTestHangDeviceOperation {
    struct tensor_args_t {
        const Tensor& tensor;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;

    using tensor_return_value_t = Tensor;

    struct operation_attributes_t {};

    struct ExecuteTestHangDeviceOperationProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };
    using program_factory_t = std::variant<ExecuteTestHangDeviceOperationProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(const Tensor& input_tensor);
};

}  // namespace ttnn::prim

namespace ttnn::operations::experimental::test {

Tensor test_hang_device_operation(const Tensor& input_tensor);

}  // namespace ttnn::operations::experimental::test
