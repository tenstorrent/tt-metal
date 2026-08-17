// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

struct SliceTileTensorArgsProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim
