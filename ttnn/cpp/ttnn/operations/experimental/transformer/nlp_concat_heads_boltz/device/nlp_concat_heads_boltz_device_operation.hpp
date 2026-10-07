// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "nlp_concat_heads_boltz_device_operation_types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct NLPConcatHeadsBoltzDeviceOperation {
    using operation_attributes_t = NLPConcatHeadsBoltzParams;
    using tensor_args_t = NLPConcatHeadsBoltzInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    // Sharded and interleaved operands share one descriptor; which one is chosen derives from the
    // memory configs, which the program hash covers. The only per-dispatch state either path has is
    // buffer addresses — runtime-arg bindings when interleaved, CB `.buffer` bindings when sharded —
    // so the framework patches both on a cache hit and no override is needed.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& output);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor nlp_concat_heads_boltz(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& memory_config,
    std::optional<Tensor> optional_output_tensor);
}  // namespace ttnn::prim
