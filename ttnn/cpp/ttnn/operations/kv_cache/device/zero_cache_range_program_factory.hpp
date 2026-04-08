// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <tt-metalium/core_coord.hpp>
#include "ttnn/device_operation.hpp"
#include "zero_cache_range_device_operation_types.hpp"

namespace ttnn::prim {

struct ZeroCacheRangeProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle writer_kernel_id = 0;
        std::vector<CoreCoord> cores;
        uint32_t g1_numcores = 0;
        uint32_t num_pages_per_core_group_1 = 0;
        uint32_t num_pages_per_core_group_2 = 0;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ZeroCacheRangeParams& operation_attributes,
        const ZeroCacheRangeInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ZeroCacheRangeParams& operation_attributes,
        const ZeroCacheRangeInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
