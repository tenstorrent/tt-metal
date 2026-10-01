// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/high_bw_all_gather/device/high_bw_all_gather_device_operation_types.hpp"

namespace ttnn::operations::experimental::fabric_all_gather {

// Same contract as high_bw_all_gather, so the same parameters (see high_bw_all_gather_device_operation_types.hpp).
using FabricAllGatherParams = high_bw_all_gather::HighBwAllGatherParams;
using FabricAllGatherInputs = high_bw_all_gather::HighBwAllGatherInputs;

}  // namespace ttnn::operations::experimental::fabric_all_gather
