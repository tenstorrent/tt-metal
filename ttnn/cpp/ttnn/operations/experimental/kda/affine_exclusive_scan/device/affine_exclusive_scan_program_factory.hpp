// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "affine_exclusive_scan_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct AffineExclusiveScanProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AffineExclusiveScanParams&, const AffineExclusiveScanInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim
