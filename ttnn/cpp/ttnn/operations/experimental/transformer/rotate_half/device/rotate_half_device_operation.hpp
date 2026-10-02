// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "rotate_half_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct RotateHalfDeviceOperation {
    using operation_attributes_t = RotateHalfParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    // Single-core program: the row/tile counts derive from the input's padded shape (hashed), so
    // the input and output buffer addresses are the only per-dispatch state and their runtime-arg
    // bindings are the whole cache-hit refresh.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& input,
        tensor_return_value_t& tensor_return_value);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor rotate_half(const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config);
}  // namespace ttnn::prim
