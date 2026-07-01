// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "rebank_rm_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct RebankRmFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RebankRmParams&      operation_attributes,
        const RebankRmTensorArgs&  tensor_args,
        ttnn::Tensor&              tensor_return_value);
};

}  // namespace ttnn::experimental::prim
