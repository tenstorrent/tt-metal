// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/quasar/untilize_with_unpadding/device/untilize_with_unpadding_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim::qsr {

struct UntilizeWithUnpaddingMultiCoreColInterleavedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output);
};

}  // namespace ttnn::prim::qsr
