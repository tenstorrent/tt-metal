// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concat_new_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

struct ConcatNewS2IProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ConcatNewParams& operation_attributes, const ConcatNewInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
