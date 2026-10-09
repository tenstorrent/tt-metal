// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concat_device_operation_types.hpp"

#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::prim::qsr {

// Quasar copy of ttnn::prim::ConcatProgramFactory, the generic TensorAccessor-based concat: per
// node, one reader DM thread gathers the output's pages from the inputs into a DFB and one writer
// DM thread drains it to the output. Any interleaved or sharded input/output whose pages are
// whole tiles (TILE) or whole rows (ROW_MAJOR) can go through it.
struct ConcatProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim::qsr
