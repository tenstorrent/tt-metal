// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "unpack_fp8_rows_device_operation_types.hpp"
#include "unpack_fp8_rows_program_factory.hpp"

namespace ttnn::operations::unpack_fp8_rows {

// Call order: the output (`out` itself) returned, the key (compute_program_hash) taken; a hit runs
// validate_on_program_cache_hit then override_runtime_arguments, a miss validate_on_program_cache_miss then
// create_descriptor.
struct UnpackFp8RowsDeviceOperation {
    using operation_attributes_t = UnpackFp8RowsParams;
    using tensor_args_t = UnpackFp8RowsInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<UnpackFp8RowsProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // Every checked tensor property is in the key and the entry checks the ranges on every call: a hit checks
    // nothing again.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // grid_x, grid_y, ncores, common_values and the rows / out specs. src_page0, out_page0 and nbands are left
    // out: override_runtime_arguments re-applies them on every hit (no arg count depends on them).
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// The Python asserts on the tensors (dtype, layout, width, placement), in their order; throws
// UnpackFp8RowsAssertion.
void check_support(const UnpackFp8RowsInputs& t);

// One structural program key, recorded once per program built (on a cache miss), for the package's _PREPARED view:
// ("unpack", mesh_id, rows_shape, out_shape, rows_mc, out_mc, ncores, (grid_x, grid_y), common_values).
struct UnpackFp8RowsPreparedKey {
    int mesh_id = 0;
    std::vector<uint32_t> rows_shape;
    std::vector<uint32_t> out_shape;
    std::string rows_mc;
    std::string out_mc;
    uint32_t ncores = 0;
    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    bool common_values = false;

    bool operator==(const UnpackFp8RowsPreparedKey&) const = default;
};

// A copy of the distinct keys recorded so far (built only when read).
std::vector<UnpackFp8RowsPreparedKey> unpack_fp8_rows_prepared_keys();

}  // namespace ttnn::operations::unpack_fp8_rows

namespace ttnn::prim {

Tensor unpack_fp8_rows(
    const Tensor& rows, const Tensor& out, const ttnn::operations::unpack_fp8_rows::UnpackFp8RowsParams& params);

}  // namespace ttnn::prim
