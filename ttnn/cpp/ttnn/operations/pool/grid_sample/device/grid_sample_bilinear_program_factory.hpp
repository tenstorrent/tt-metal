// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/pool/grid_sample/device/grid_sample_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct GridSampleBilinearProgramFactory {
    struct shared_variables_t {
        bool is_sharded = false;
        std::vector<tt::tt_metal::CoreCoord> logical_cores;
        tt::tt_metal::CBHandle grid_cb_handle{};
        tt::tt_metal::CBHandle output_cb_handle{};
        uint32_t num_cores = 0;
        bool enable_split_reader = false;
        tt::tt_metal::KernelHandle reader0_kernel_id{};
        tt::tt_metal::KernelHandle reader1_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const GridSampleParams& operation_attributes, const GridSampleInputs& tensor_args, Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const GridSampleParams& operation_attributes,
        const GridSampleInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
