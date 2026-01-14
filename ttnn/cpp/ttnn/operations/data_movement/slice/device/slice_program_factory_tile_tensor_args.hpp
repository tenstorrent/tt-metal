// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::data_movement::slice::program {

struct SliceTileTensorArgsProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle unary_reader_kernel_id{};
        tt::tt_metal::KernelHandle unary_writer_kernel_id{};
        CoreCoord compute_with_storage_grid_size;
        std::optional<CoreRangeSet> sub_core_grids;
        std::vector<uint32_t> accumulated_total_per_dim;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& args, const tensor_args_t& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& args,
        const tensor_args_t& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::operations::data_movement::slice::program
