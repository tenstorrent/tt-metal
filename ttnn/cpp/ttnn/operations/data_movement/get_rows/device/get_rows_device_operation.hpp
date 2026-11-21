// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include "get_rows_device_operation_types.hpp"
#include "get_rows_program_factory.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

// Device-side row gather: output[i] = input[indices[i]] for a [H, W] row-major input and an [N]
// index vector. Equivalent to torch.index_select(input, 0, indices) for non-negative in-range
// indices; the host wrapper rejects anything else before dispatch so no OOB NoC read can happen.
struct GetRowsDeviceOperation {
    using operation_attributes_t = GetRowsParams;
    using tensor_args_t = GetRowsInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = GetRowsProgramFactory;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor get_rows(
    const Tensor& input_tensor,
    const Tensor& indices_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

}  // namespace ttnn::prim
