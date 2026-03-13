// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "dit_minimal_rm_binary_device_operation_types.hpp"
#include "dit_minimal_rm_binary_program_factory.hpp"

namespace ttnn::experimental::prim {

struct DitMinimalRmBinaryDeviceOperation {
    using operation_attributes_t = DitMinimalRmBinaryParams;
    using tensor_args_t = DitMinimalRmBinaryInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<DitMinimalRmBinaryProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor dit_minimal_rm_binary(
    const Tensor& input_a,
    const Tensor& input_b,
    ttnn::experimental::prim::BinaryOpType op_type,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& output_memory_config);

}  // namespace ttnn::prim
