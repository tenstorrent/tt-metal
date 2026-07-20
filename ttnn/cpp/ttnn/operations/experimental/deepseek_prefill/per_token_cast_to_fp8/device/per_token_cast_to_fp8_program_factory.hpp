// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "per_token_cast_to_fp8_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim::per_token_cast_to_fp8 {

struct PerTokenCastToFp8SharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    std::vector<CoreCoord> all_cores_vec;
};

struct PerTokenCastToFp8ProgramFactory {
    using shared_variables_t = PerTokenCastToFp8SharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;

    static cached_program_t create(
        const PerTokenCastToFp8Params& operation_attributes,
        const PerTokenCastToFp8Inputs& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const PerTokenCastToFp8Params& operation_attributes,
        const PerTokenCastToFp8Inputs& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

// TILE-layout input variant. Input tiles are read directly (no on-core tilize); a block is the
// tiles_per_block tiles covering one [tile_h x 128] region, so the per-row amax maps to one logical
// token row. Both outputs stay ROW_MAJOR, so the compute still untilizes the e4m3 result.
struct PerTokenCastToFp8TileProgramFactory {
    using shared_variables_t = PerTokenCastToFp8SharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;

    static cached_program_t create(
        const PerTokenCastToFp8Params& operation_attributes,
        const PerTokenCastToFp8Inputs& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const PerTokenCastToFp8Params& operation_attributes,
        const PerTokenCastToFp8Inputs& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::experimental::prim::per_token_cast_to_fp8
