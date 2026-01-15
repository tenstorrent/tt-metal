// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "reshard_device_operation_types.hpp"

namespace ttnn::prim {

// Factory for DRAM->DRAM nd reshard (simple page by page copy)
struct NdReshardCopyPagesFactory {
    struct NdReshardCopyPagesSharedVariables {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        CoreRangeSet grid;
        std::vector<CoreCoord> cores;
    };

    using shared_variables_t = NdReshardCopyPagesSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ReshardParams& operation_attributes, const ReshardInputs& tensor_args, Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ReshardParams& operation_attributes,
        const ReshardInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
