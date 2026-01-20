// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/host_api.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim {

struct SliceRmProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle unary_reader_kernel_id{};
        tt::tt_metal::KernelHandle unary_writer_kernel_id{};
        CoreCoord compute_with_storage_grid_size;
        std::optional<CoreRangeSet> sub_core_grids;
        tt::tt_metal::CBHandle cb_src0{};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program, const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim
