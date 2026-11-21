// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::llama31_8b_v3 {

struct SwigluDecodeParams {};

struct SwigluDecodeInputs {
    Tensor gate;
    Tensor up;
};

struct SwigluDecodeProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SwigluDecodeParams& operation_attributes, const SwigluDecodeInputs& tensor_args, Tensor& output);
};

struct SwigluDecodeOperation {
    using operation_attributes_t = SwigluDecodeParams;
    using tensor_args_t = SwigluDecodeInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<SwigluDecodeProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

bool swiglu_decode_supported(const Tensor& gate, const Tensor& up);

}  // namespace ttnn::experimental::llama31_8b_v3

namespace ttnn::prim {
ttnn::Tensor llama31_8b_v3_swiglu_decode(const Tensor& gate, const Tensor& up);
}  // namespace ttnn::prim
