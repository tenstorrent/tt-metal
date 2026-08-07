// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

namespace ttnn::for_python {

// Metal 1.0 copy of ttnn::prim::SliceTileProgramFactory, frozen for the Python fusion framework.
// Not part of SliceDeviceOperation and not on any dispatch path — do not port it to Metal 2.0.
struct SliceTileProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ttnn::prim::SliceParams& args, const ttnn::prim::SliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::for_python
