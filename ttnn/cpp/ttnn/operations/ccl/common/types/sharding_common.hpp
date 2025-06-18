// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// File contains enumerations that are common to both kernel and program factories with regards to sharding

#pragma once

#include <tt-metalium/buffer_types.hpp>

namespace shard_addr_gen_consts {

enum class ContiguityType {
    // Indicates logical sharding placed padding between pages so no contiguous pages exist
    L1_PADDING_BETWEEN_PAGES = 0,
    // Indicates some padding exists in the rightmost shard since the pages did not divide evenly into shards
    L1_PADDING_IN_RIGHTMOST_SHARD,
    // Indicates no sharding based padding exists so all pages within the same shard are contiguous
    // This is useful for height sharded tensors as multiple rows of the tensor can be contiguous.
    L1_NO_SHARD_PADDING,

    // DRAM variants
    DRAM_PADDING_BETWEEN_PAGES,
    DRAM_PADDING_IN_RIGHTMOST_SHARD,
    DRAM_NO_SHARD_PADDING,
};

}  // namespace shard_addr_gen_consts
