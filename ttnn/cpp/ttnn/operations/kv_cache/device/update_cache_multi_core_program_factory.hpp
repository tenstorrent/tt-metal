// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "update_cache_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::kv_cache::program {

struct UpdateCacheMultiCoreProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle unary_reader_kernel_id {};
        tt::tt_metal::KernelHandle unary_writer_kernel_id {};
        std::vector<CoreCoord> cores;
        uint32_t Wbytes = 0;
        uint32_t Wt = 0;
        std::vector<uint32_t> cache_start_ids;
        tt::tt_metal::CBHandle cb_src1 {};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::operations::kv_cache::program
