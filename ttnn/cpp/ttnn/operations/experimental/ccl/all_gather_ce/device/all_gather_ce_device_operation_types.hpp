// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// Fork of all_gather_async types for an independent device op / program cache entry.
using AllGatherCeParams = AllGatherAsyncParams;
using AllGatherCeInputs = AllGatherAsyncInputs;

}  // namespace ttnn::experimental::prim
