// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "scatter_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::data_movement::scatter {

using namespace tt;
using namespace tt::tt_metal;

struct ScatterProgramFactory {
    struct shared_variables_t {
        KernelHandle reader_kernel_id{};
        KernelHandle writer_kernel_id{};
        std::vector<CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const operation_attributes_t&, const tensor_args_t&, Tensor&);

    static void override_runtime_arguments(
        cached_program_t&, const operation_attributes_t&, const tensor_args_t&, Tensor&);
};

}  // namespace ttnn::operations::data_movement::scatter
