// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "typecast_device_op_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::prim {

// Typecast fused with a TILE <-> ROW_MAJOR layout change in one pass.
struct TypecastLayoutChangeProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output);
};

// Whether the fused factory handles this input / output pair; callers fall back to separate ops otherwise.
bool can_fuse_typecast_layout_change(
    const Tensor& input,
    const tt::tt_metal::TensorSpec& output_spec,
    const std::optional<CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::prim
