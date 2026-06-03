// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "zero_cache_range_device_operation_types.hpp"

namespace ttnn::prim {

struct ZeroCacheRangeProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ZeroCacheRangeParams& operation_attributes,
        const ZeroCacheRangeInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
