// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/metal_v2_artifacts.hpp"

#include "concat_heads_decode_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct ConcatHeadsDecodeSubcoregridsProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ConcatHeadsDecodeParams& operation_attributes,
        const ConcatHeadsDecodeInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::experimental::prim
