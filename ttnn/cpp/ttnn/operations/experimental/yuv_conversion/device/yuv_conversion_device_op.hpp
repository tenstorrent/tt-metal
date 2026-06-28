// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "yuv_conversion_device_op_types.hpp"
#include "yuv_conversion_program_factory.hpp"

namespace ttnn::experimental::prim {

struct YUVConversionDeviceOperation {
    using operation_attributes_t = YUVConversionParams;
    using tensor_args_t = YUVConversionInputs;
    using spec_return_value_t = std::tuple<TensorSpec, TensorSpec, TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor, Tensor>;
    using program_factory_t = std::variant<YUVConversionProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::tuple<Tensor, Tensor, Tensor> yuv_conversion(
    const Tensor& input,
    const ttnn::experimental::prim::YUVCoefficients& coefficients,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::prim
