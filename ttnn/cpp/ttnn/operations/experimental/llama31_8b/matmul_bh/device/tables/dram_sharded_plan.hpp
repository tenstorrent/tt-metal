// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/factory/dram_sharded.hpp"

namespace ttnn::prim::llama31_8b_matmul_bh::dram_sharded {

// clang-format off
inline constexpr DsPlanReader kDsReadersLmHeadDec[] = {{0, 36}, {36, 36}, {0, 36}, {36, 36}, {0, 36}, {36, 36}, {0, 36}, {36, 36}, {0, 36}, {36, 36}, {0, 36}, {36, 36}, {0, 36}, {36, 36}};
inline constexpr uint8_t kDsSlotsLmHeadDec[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
inline constexpr DramShardedPlan kDsPlanLmHeadDec = {
    .Nt = 501,
    .Kt = 128,
    .in0_block_w = 2,
    .workers_per_bank = 2,
    .num_readers = 14,
    .in1_shard_width_tiles = 72,
    .in0_storage_cores = 64,
    .per_core_N_in1_sender = 36,
    .per_core_N_compute = 40,
    .out_subblock_h = 1,
    .out_subblock_w = 8,
    .last_subblock_w_valid = 4,
    .in1_num_subblocks = 5,
    .in0_num_subblocks = 1,
    .num_blocks = 64,
    .per_core_N_storage = 8,
    .num_cores_written_back = 63,
    .readers = kDsReadersLmHeadDec,
    .in0_sender_slots = kDsSlotsLmHeadDec};

inline constexpr DsPlanReader kDsReadersFf13Dec[] = {{0, 32}, {32, 32}, {0, 32}, {32, 32}, {0, 32}, {32, 32}, {0, 32}, {32, 32}, {0, 32}, {32, 32}, {0, 32}, {32, 32}, {0, 32}, {32, 32}};
inline constexpr uint8_t kDsSlotsFf13Dec[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
inline constexpr DramShardedPlan kDsPlanFf13Dec = {
    .Nt = 448,
    .Kt = 128,
    .in0_block_w = 2,
    .workers_per_bank = 2,
    .num_readers = 14,
    .in1_shard_width_tiles = 64,
    .in0_storage_cores = 64,
    .per_core_N_in1_sender = 32,
    .per_core_N_compute = 32,
    .out_subblock_h = 1,
    .out_subblock_w = 8,
    .last_subblock_w_valid = 8,
    .in1_num_subblocks = 4,
    .in0_num_subblocks = 1,
    .num_blocks = 64,
    .per_core_N_storage = 7,
    .num_cores_written_back = 64,
    .readers = kDsReadersFf13Dec,
    .in0_sender_slots = kDsSlotsFf13Dec};

inline constexpr DsPlanReader kDsReadersFf2Dec[] = {{0, 10}, {10, 9}, {0, 10}, {10, 9}, {0, 10}, {10, 9}, {0, 10}, {10, 9}, {0, 10}, {10, 9}, {0, 10}, {10, 9}, {0, 10}, {10, 9}};
inline constexpr uint8_t kDsSlotsFf2Dec[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
inline constexpr DramShardedPlan kDsPlanFf2Dec = {
    .Nt = 128,
    .Kt = 448,
    .in0_block_w = 7,
    .workers_per_bank = 2,
    .num_readers = 14,
    .in1_shard_width_tiles = 19,
    .in0_storage_cores = 64,
    .per_core_N_in1_sender = 10,
    .per_core_N_compute = 10,
    .out_subblock_h = 1,
    .out_subblock_w = 5,
    .last_subblock_w_valid = 5,
    .in1_num_subblocks = 2,
    .in0_num_subblocks = 1,
    .num_blocks = 64,
    .per_core_N_storage = 2,
    .num_cores_written_back = 64,
    .readers = kDsReadersFf2Dec,
    .in0_sender_slots = kDsSlotsFf2Dec};

inline constexpr DsPlanReader kDsReadersQkvDec[] = {{0, 10}, {10, 10}, {20, 8}, {0, 10}, {10, 10}, {20, 8}, {0, 10}, {10, 10}, {20, 8}, {0, 10}, {10, 10}, {20, 8}, {0, 10}, {10, 10}, {20, 8}, {0, 10}, {10, 10}, {20, 8}, {0, 10}, {10, 10}, {20, 8}};
inline constexpr uint8_t kDsSlotsQkvDec[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
inline constexpr DramShardedPlan kDsPlanQkvDec = {
    .Nt = 192,
    .Kt = 128,
    .in0_block_w = 4,
    .workers_per_bank = 3,
    .num_readers = 21,
    .in1_shard_width_tiles = 28,
    .in0_storage_cores = 32,
    .per_core_N_in1_sender = 10,
    .per_core_N_compute = 12,
    .out_subblock_h = 1,
    .out_subblock_w = 4,
    .last_subblock_w_valid = 2,
    .in1_num_subblocks = 3,
    .in0_num_subblocks = 1,
    .num_blocks = 32,
    .per_core_N_storage = 6,
    .num_cores_written_back = 32,
    .readers = kDsReadersQkvDec,
    .in0_sender_slots = kDsSlotsQkvDec};

inline constexpr DsPlanReader kDsReadersWoDec[] = {{0, 7}, {7, 7}, {14, 5}, {0, 7}, {7, 7}, {14, 5}, {0, 7}, {7, 7}, {14, 5}, {0, 7}, {7, 7}, {14, 5}, {0, 7}, {7, 7}, {14, 5}, {0, 7}, {7, 7}, {14, 5}, {0, 7}, {7, 7}, {14, 5}};
inline constexpr uint8_t kDsSlotsWoDec[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
inline constexpr DramShardedPlan kDsPlanWoDec = {
    .Nt = 128,
    .Kt = 128,
    .in0_block_w = 4,
    .workers_per_bank = 3,
    .num_readers = 21,
    .in1_shard_width_tiles = 19,
    .in0_storage_cores = 32,
    .per_core_N_in1_sender = 7,
    .per_core_N_compute = 8,
    .out_subblock_h = 1,
    .out_subblock_w = 4,
    .last_subblock_w_valid = 3,
    .in1_num_subblocks = 2,
    .in0_num_subblocks = 1,
    .num_blocks = 32,
    .per_core_N_storage = 4,
    .num_cores_written_back = 32,
    .readers = kDsReadersWoDec,
    .in0_sender_slots = kDsSlotsWoDec};

// clang-format on

}  // namespace ttnn::prim::llama31_8b_matmul_bh::dram_sharded
