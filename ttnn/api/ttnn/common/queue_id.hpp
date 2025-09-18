// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt_stl/strong_type.hpp>
#include <cstdint>

namespace ttnn {
/*
    Type must be moved to metal.
    Background:
    We have two software command queues available to overlap some work and reduce latency.
    For example, Op2 can be prepared in a different queue while the first queue is blocked, waiting for data readout by
    Op1. TT-NN operations allow specifying which queue should be used. The default queue is 0, and the possible values
    are 0 and 1.
*/
using QueueId = tt::stl::StrongType<uint8_t, struct QueueIdTag>;

}  // namespace ttnn

// Exporting to tt::tt_metal namespace because ttnn
// defines some of its own types (think Tensor) in tt::tt_metal namespace.
namespace tt::tt_metal {

using QueueId = ttnn::QueueId;

inline std::optional<uint8_t> raw_optional(const std::optional<QueueId>& cq_id) {
    return cq_id.has_value() ? std::make_optional(cq_id.value().get()) : std::nullopt;
}

}  // namespace tt::tt_metal
