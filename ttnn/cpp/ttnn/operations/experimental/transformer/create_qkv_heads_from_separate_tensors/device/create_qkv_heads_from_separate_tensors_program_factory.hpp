// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "create_qkv_heads_from_separate_tensors_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct CreateQKVHeadsSeparateTensorsProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const CreateQKVHeadsFromSeparateTensorsParams& operation_attributes,
        const CreateQKVHeadsFromSeparateTensorsInputs& tensor_args,
        CreateQKVHeadsFromSeparateTensorsResult& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
