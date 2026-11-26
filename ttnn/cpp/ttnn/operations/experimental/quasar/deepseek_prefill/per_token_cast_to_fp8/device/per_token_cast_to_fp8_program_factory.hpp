// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "per_token_cast_to_fp8_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/workload/legacy/program_descriptors.hpp>

namespace ttnn::experimental::prim::qsr::per_token_cast_to_fp8 {

// Single factory for both input layouts. create_descriptor() reads the input layout and selects the ROW_MAJOR vs
// TILE work-split / kernel define internally (compute skips input tilization only for TILE); everything
// else is shared. ROW_MAJOR and TILE stay distinct program-cache entries because the default program hash
// covers the input's TensorSpec (which includes its layout).
//
// No override_runtime_arguments: only the I/O buffer addresses vary between two calls sharing a cache key, and
// they are declared as Buffer* bindings, which the framework re-points on a cache hit.
struct PerTokenCastToFp8ProgramFactory {
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PerTokenCastToFp8Params& operation_attributes,
        const PerTokenCastToFp8Inputs& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::experimental::prim::qsr::per_token_cast_to_fp8
