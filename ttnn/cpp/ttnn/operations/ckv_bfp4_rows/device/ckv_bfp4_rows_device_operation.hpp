// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "ckv_bfp4_rows_device_operation_types.hpp"
#include "ckv_bfp4_rows_program_factory.hpp"

namespace ttnn::operations::ckv_bfp4_rows {

// Framework call order: create_output_tensors (returns `rows`), compute_program_hash, then on a hit
// validate_on_program_cache_hit + override_runtime_arguments, on a miss validate_on_program_cache_miss +
// create_descriptor.
struct CkvBfp4RowsDeviceOperation {
    using operation_attributes_t = CkvBfp4RowsParams;
    using tensor_args_t = CkvBfp4RowsInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<CkvBfp4RowsProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // The key fixes geometry and grid; a hit re-checks the per-call row ranges and the preallocated output.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// The Python entry's asserts, in order; throws ContractAssertion with the Python's messages.
void check_support(const Tensor& src, const Tensor& rows, uint32_t src_row0, uint32_t n, uint32_t row0);

}  // namespace ttnn::operations::ckv_bfp4_rows

namespace ttnn::prim {

Tensor ckv_bfp4_rows(
    const Tensor& src,
    const Tensor& rows,
    const std::optional<Tensor>& output,
    const ttnn::operations::ckv_bfp4_rows::CkvBfp4RowsParams& params);

}  // namespace ttnn::prim
