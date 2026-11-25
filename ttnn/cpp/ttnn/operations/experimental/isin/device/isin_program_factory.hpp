// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "isin_device_operation_types.hpp"

#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {
struct IsInProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const IsinParams&, const IsinInputs&, Tensor&);
};

}  // namespace ttnn::experimental::prim
