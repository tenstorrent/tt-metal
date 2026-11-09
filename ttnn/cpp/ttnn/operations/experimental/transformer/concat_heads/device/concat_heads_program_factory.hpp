// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concat_heads_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct ConcatHeadsProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ConcatHeadsParams& operation_attributes, const Tensor& input, Tensor& output);
};

}  // namespace ttnn::experimental::prim
