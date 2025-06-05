// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/work_split.hpp>

#include "../scatter_enums.hpp"
#include "scatter_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::scatter {

enum class ScatterCB : std::underlying_type_t<tt::CBIndex> {
    INPUT = CBIndex::c_0,
    SRC = CBIndex::c_1,
    INDEX = CBIndex::c_2,
    DST = CBIndex::c_3
};

using namespace tt;
using namespace tt::tt_metal;

struct ScatterProgramFactory {
    struct shared_variables_t {
        KernelHandle reader_kernel_id;
        KernelHandle writer_kernel_id;
        CoreCoord storage_grid_size;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);

    static void override_runtime_arguments(
        cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);

    static CBHandle create_cb(
        Program& program,
        const DataType& dtype,
        const ScatterCB& scatter_cb,
        const CoreRangeSet& core_range_set,
        const uint32_t& tiles_num);

    static KernelHandle create_kernel(
        Program& program,
        const char* kernel_path,
        const CoreRangeSet& core_range_set,
        const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config,
        const std::vector<uint32_t>& runtime_args = {});
};

}  // namespace ttnn::operations::experimental::scatter
