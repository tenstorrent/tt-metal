// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::llama31_8b {

struct ArgmaxDecodeParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct ArgmaxDecodeInputs {
    Tensor input;
    std::optional<Tensor> optional_output_tensor;
};

struct ArgmaxDecodeProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ArgmaxDecodeParams& operation_attributes, const ArgmaxDecodeInputs& tensor_args, Tensor& output);
};

struct ArgmaxDecodeOperation {
    using operation_attributes_t = ArgmaxDecodeParams;
    using tensor_args_t = ArgmaxDecodeInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<ArgmaxDecodeProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

namespace argmax_decode_table {
struct Shape;
}

const argmax_decode_table::Shape* argmax_decode_shape(const Tensor& input);

bool argmax_decode_supported(
    const Tensor& input,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& output_tensor);

}  // namespace ttnn::experimental::llama31_8b

namespace ttnn::prim {
ttnn::Tensor llama31_8b_argmax_decode(
    const Tensor& input,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<Tensor>& optional_output_tensor);
}  // namespace ttnn::prim
