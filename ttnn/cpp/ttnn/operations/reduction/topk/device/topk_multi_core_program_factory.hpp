// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/reduction/topk/device/topk_device_operation_types.hpp"

namespace ttnn::prim {

struct TopKMultiCoreSharedVariables {
    tt::tt_metal::KernelHandle unary_reader_kernel_id{};
    tt::tt_metal::KernelHandle binary_writer_final_kernel_id{};
    std::vector<tt::tt_metal::CoreCoord> local_cores;
    tt::tt_metal::CoreCoord final_core;
};

struct TopKMultiCoreProgramFactory {
    using shared_variables_t = TopKMultiCoreSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const TopkParams& args, const TopkInputs& tensor_args, std::tuple<Tensor, Tensor>& output_tensors);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const TopkParams& args,
        const TopkInputs& tensor_args,
        std::tuple<Tensor, Tensor>& output_tensors);
};

}  // namespace ttnn::prim
