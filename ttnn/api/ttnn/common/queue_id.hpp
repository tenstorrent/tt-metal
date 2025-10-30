// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/queue_id.hpp>

namespace ttnn {
/*
    Moved to metal as a part of tensor lowering. ttnn::QueueId remains to avoid breaking existing code and users, but
   all new development should use tt::tt_metal::QueueId instead.
*/
using QueueId = tt::tt_metal::QueueId;

}  // namespace ttnn
