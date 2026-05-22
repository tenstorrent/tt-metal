// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "plusone_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct PlusOneProgramFactory {
    // Contract (1): per-coord ProgramDescriptor.  Single reader kernel that
    // increments every entry of the input tensor (in-place semantics).  When
    // the input is sharded the CB is bound via .buffer for dynamic CB address
    // re-application; otherwise the buffer base address is carried via the
    // per-core runtime arg.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PlusoneParams& operation_attributes, const Tensor& input, Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
