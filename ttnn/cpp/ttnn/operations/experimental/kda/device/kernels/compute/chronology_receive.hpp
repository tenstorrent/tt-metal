// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"

namespace kda_chronology {

// Wait for the topology tile at the front of `buffer`, read its eight words, and pop it. Compute kernels only:
// DataflowBuffer::read_tile_value is declared for TRISC alone.
inline Topology receive(DataflowBuffer& buffer) {
    buffer.wait_front(1);
    Topology result{
        buffer.read_tile_value(0, 0),
        buffer.read_tile_value(0, 1),
        buffer.read_tile_value(0, 2),
        buffer.read_tile_value(0, 3),
        buffer.read_tile_value(0, 4),
        buffer.read_tile_value(0, 5),
        buffer.read_tile_value(0, 6),
        buffer.read_tile_value(0, 7)};
    buffer.pop_front(1);
    return result;
}

}  // namespace kda_chronology
