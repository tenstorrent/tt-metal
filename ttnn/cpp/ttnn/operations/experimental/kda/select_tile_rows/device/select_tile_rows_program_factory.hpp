// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "select_tile_rows_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct SelectTileRowsProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SelectTileRowsParams&, const SelectTileRowsInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim
