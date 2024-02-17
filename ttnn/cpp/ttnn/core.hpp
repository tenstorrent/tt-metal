// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tensor/types.hpp"

namespace ttnn {
namespace core {

static const auto DRAM_MEMORY_CONFIG = tt::tt_metal::MemoryConfig{
    .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED, .buffer_type = tt::tt_metal::BufferType::DRAM};
static const auto L1_MEMORY_CONFIG = tt::tt_metal::MemoryConfig{
    .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED, .buffer_type = tt::tt_metal::BufferType::L1};

}  // namespace core

using core::DRAM_MEMORY_CONFIG;
using core::L1_MEMORY_CONFIG;

}  // namespace ttnn
