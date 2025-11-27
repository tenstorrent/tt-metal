// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "scatter_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::data_movement::scatter {

enum class ScatterReduceBfloat16CB : std::underlying_type_t<tt::CBIndex> {
    INPUT = CBIndex::c_0,
    SRC = CBIndex::c_1,
    INDEX = CBIndex::c_2,
    DST = CBIndex::c_3,
    FP32_TEMP = CBIndex::c_4
};

using namespace tt;
using namespace tt::tt_metal;

struct ScatterReduceBfloat16ProgramFactory {
    struct shared_variables_t {
        KernelHandle reader_kernel_id{};
        KernelHandle writer_kernel_id{};
        std::vector<CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);

    static void override_runtime_arguments(
        cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};

}  // namespace ttnn::operations::data_movement::scatter
