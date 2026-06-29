// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/quasar/untilize_with_unpadding/device/untilize_with_unpadding_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/program_spec_artifacts.hpp"

namespace ttnn::prim::qsr {

struct UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory {
    static ttnn::device_operation::ProgramSpecArtifacts create_program_spec(
        const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output);
};

}  // namespace ttnn::prim::qsr
