// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/queue_id.hpp>

namespace ttnn {
/*
    Moved to metal as a part of tensor lowering.
*/
using QueueId = tt::tt_metal::QueueId;

}  // namespace ttnn
