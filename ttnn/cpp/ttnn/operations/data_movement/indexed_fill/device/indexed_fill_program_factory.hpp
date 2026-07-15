// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct IndexedFillProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const IndexedFillParams& operation_attributes, const IndexedFillInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim
