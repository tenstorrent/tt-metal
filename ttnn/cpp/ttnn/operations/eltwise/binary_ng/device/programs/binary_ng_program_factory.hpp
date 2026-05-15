// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation_types.hpp"

namespace ttnn::operations::binary_ng::program {

struct BinaryNgProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const BinaryNgParams& operation_attributes, const BinaryNgInputs& tensor_args, Tensor& c);
};
}  // namespace ttnn::operations::binary_ng::program
