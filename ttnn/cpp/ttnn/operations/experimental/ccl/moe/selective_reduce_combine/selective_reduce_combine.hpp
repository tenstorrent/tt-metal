// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "device/selective_reduce_combine_device_operation.hpp"

namespace ttnn::experimental {

using ttnn::prim::selective_reduce_combine;

}  // namespace ttnn::experimental
