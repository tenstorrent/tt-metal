// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "create_qkv_heads_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct CreateQKVHeadsProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const CreateQKVHeadsParams& operation_attributes,
        const CreateQKVHeadsInputs& tensor_args,
        CreateQKVHeadsResult& output);
};

}  // namespace ttnn::experimental::prim
