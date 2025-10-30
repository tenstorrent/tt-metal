// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "intimg_device_operation_types.hpp"

#include <cstdint>

#include "intimg_program_factory.hpp"

#include <optional>
#include <type_traits>
#include <variant>

#include "hostdevcommon/kernel_structs.h"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::reduction {

using namespace tt::tt_metal;
using namespace tt::stl;

struct IntImgDeviceOperation {
    using operation_attributes_t = reduction::operation_attributes_t;
    using tensor_args_t = reduction::tensor_args_t;
    using spec_return_value_t = reduction::spec_return_value_t;
    using tensor_return_value_t = reduction::tensor_return_value_t;
    using program_factory_t = std::variant<reduction::IntImgProgramFactory>;

    using invocation_result_t = std::tuple<operation_attributes_t, tensor_args_t>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static invocation_result_t invoke(const Tensor& input_tensor);
};

}  // namespace ttnn::operations::experimental::reduction

namespace ttnn::prim {
constexpr auto intimg =
    ttnn::register_operation<"ttnn::prim::intimg", ttnn::operations::experimental::reduction::IntImgDeviceOperation>();
}  // namespace ttnn::prim
