// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Device-op skeleton for ttnn::prim::rebank_rm — page-size-converting
// copy for ROW_MAJOR fp32/bf16 tensors.

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

#include "rebank_rm_device_operation_types.hpp"
#include "rebank_rm_factory.hpp"

namespace ttnn::experimental::prim {

struct RebankRmDeviceOperation {
    using operation_attributes_t = RebankRmParams;
    using tensor_args_t          = RebankRmTensorArgs;

    using spec_return_value_t   = TensorSpec;
    using tensor_return_value_t = Tensor;

    using program_factory_t = std::variant<RebankRmFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(
        const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Public entry point.  Re-pages input (B_total, N) → (B_total*N/chunk, chunk).
// Requires: chunk is a pow-2 divisor of N; fp32 or bf16; ROW_MAJOR.
Tensor rebank_rm(const Tensor& input, uint32_t chunk_size);

}  // namespace ttnn::prim
