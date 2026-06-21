// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "deltanet_prefill_chunked_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deltanet {

struct DeltaNetPrefillChunkedSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id;
    tt::tt_metal::KernelHandle compute_kernel_id;
    tt::tt_metal::KernelHandle writer_kernel_id;
    CoreRangeSet all_cores;
};

struct DeltaNetPrefillChunkedProgramFactory {
    using shared_variables_t = DeltaNetPrefillChunkedSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    using operation_attributes_t = DeltaNetPrefillChunkedParams;
    using tensor_args_t = DeltaNetPrefillChunkedInputs;
    using tensor_return_value_t = std::vector<Tensor>;

    static cached_program_t create(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void override_runtime_arguments(
        cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};

}  // namespace ttnn::operations::experimental::deltanet
