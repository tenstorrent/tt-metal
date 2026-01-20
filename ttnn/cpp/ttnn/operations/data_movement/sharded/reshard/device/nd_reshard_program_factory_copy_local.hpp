// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "reshard_device_operation_types.hpp"

namespace ttnn::prim {

// Factory for L1<->DRAM or L1->L1 nd reshard (read into local pages in L1)
template <bool local_is_input>
struct NdReshardCopyLocalShardFactory {
    struct NdReshardCopyLocalShardSharedVariables {
        tt::tt_metal::KernelHandle brisc_kernel_id{};
        tt::tt_metal::KernelHandle ncrisc_kernel_id{};
        CoreRangeSet grid;
        std::vector<CoreCoord> cores;
    };

    using shared_variables_t = NdReshardCopyLocalShardSharedVariables;
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
