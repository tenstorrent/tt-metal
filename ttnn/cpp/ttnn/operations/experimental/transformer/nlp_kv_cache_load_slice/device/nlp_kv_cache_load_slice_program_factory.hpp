// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "nlp_kv_cache_load_slice_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct NlpKVCacheLoadSliceProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const NlpKvCacheLoadSliceParams& operation_attributes,
        const NlpKvCacheLoadSliceInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::experimental::prim
