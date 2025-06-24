// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "compute_kernel_api.h"

namespace NAMESPACE {

void MAIN { DPRINT_MATH(DPRINT << "Not hanging the device, use this only for testing!!!" << ENDL()); }

}  // namespace NAMESPACE
