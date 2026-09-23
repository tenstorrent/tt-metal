// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "isin_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {
struct IsInProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(const IsinParams&, const IsinInputs&, Tensor&);
};

}  // namespace ttnn::experimental::prim
