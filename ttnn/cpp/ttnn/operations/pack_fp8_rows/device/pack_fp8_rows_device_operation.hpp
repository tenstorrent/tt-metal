// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "pack_fp8_rows_device_operation_types.hpp"
#include "pack_fp8_rows_program_factory.hpp"

namespace ttnn::operations::pack_fp8_rows {

// Call order: outputs created, the key (compute_program_hash) taken; a hit runs
// validate_on_program_cache_hit then override_runtime_arguments, a miss validate_on_program_cache_miss then
// create_descriptor.
struct PackFp8RowsDeviceOperation {
    using operation_attributes_t = PackFp8RowsParams;
    using tensor_args_t = PackFp8RowsInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<PackFp8RowsProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // Every checked input property (dtypes, layouts, shapes, memory configs) is in the key, and the public entry
    // checks a caller's output on every call, so a hit checks nothing again.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // The attributes and the e4m3 / scale specs. Whether the output is the caller's or allocated is left out:
    // either way its spec follows from those (the entry refuses any other), so both share one program.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// The Python asserts, in their order; throws PackFp8RowsAssertion.
void check_support(const PackFp8RowsParams& attrs, const PackFp8RowsInputs& t);

}  // namespace ttnn::operations::pack_fp8_rows

namespace ttnn::prim {

Tensor pack_fp8_rows(
    const Tensor& e4m3,
    const Tensor& scale,
    const std::optional<Tensor>& output,
    const ttnn::operations::pack_fp8_rows::PackFp8RowsParams& params);

}  // namespace ttnn::prim
