// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nlp_kv_cache_load_slice_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct NlpKVCacheLoadSliceSharedVariables {
    tt::tt_metal::KernelHandle unary_reader_kernel_id{};
    tt::tt_metal::KernelHandle unary_writer_kernel_id{};
    tt::tt_metal::CBHandle cb_src0{};
    uint32_t num_cores_total = 0;
    uint32_t num_cores_x = 0;
};

struct NlpKVCacheLoadSliceProgramFactory {
    using shared_variables_t = NlpKVCacheLoadSliceSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const NlpKvCacheLoadSliceParams& operation_attributes,
        const NlpKvCacheLoadSliceInputs& tensor_args,
        Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const NlpKvCacheLoadSliceParams& operation_attributes,
        const NlpKvCacheLoadSliceInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::experimental::prim
