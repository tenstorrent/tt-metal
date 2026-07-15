// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "prefix_scan_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct PrefixScanProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PrefixScanParams& operation_attributes, const PrefixScanInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
