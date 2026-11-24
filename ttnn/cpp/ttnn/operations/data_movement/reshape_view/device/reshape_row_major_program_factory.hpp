// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/reshape_device_operation_types.hpp"

namespace ttnn::prim {

// Depth of the RM factory's destination staging ring. Depends on live L1 occupancy when pages are
// not NoC-aligned, so compute_program_hash must key on it; otherwise a cache hit replays a slot
// count budgeted against a different L1 state.
uint32_t reshape_rm_num_dest_write_slots(
    const ReshapeViewParams& operation_attributes, const ReshapeViewInputs& tensor_args);

struct ReshapeViewRMProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ReshapeViewParams& operation_attributes,
        const ReshapeViewInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
