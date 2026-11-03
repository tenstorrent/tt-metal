// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dequantize_mxfp8_program_factory.hpp"

namespace ttnn::experimental::prim {
struct Mxfp8DeviceOperation {
    using operation_attributes_t = Mxfp8Params;
    using tensor_args_t = Mxfp8Inputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<Mxfp8ProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::experimental::prim
