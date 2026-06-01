// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "deltanet_prefill_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deltanet {

struct DeltaNetPrefillFullSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id;
    tt::tt_metal::KernelHandle compute_kernel_id;
    tt::tt_metal::KernelHandle writer_kernel_id;
    CoreRangeSet all_cores;
};

struct DeltaNetPrefillFullProgramFactory {
    using shared_variables_t = DeltaNetPrefillFullSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    using operation_attributes_t = DeltaNetPrefillFullParams;
    using tensor_args_t = DeltaNetPrefillFullInputs;
    using tensor_return_value_t = std::vector<Tensor>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deltanet
