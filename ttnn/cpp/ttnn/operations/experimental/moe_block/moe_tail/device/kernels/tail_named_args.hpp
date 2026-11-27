// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE tail kernels' named compile-time args: one name for each meaning, shared by the host
// (moe_tail_program_factory.cpp: `set_named_compile_time_arg(kernel, moe_tail_named_args::kName, value)`) and the
// kernels (`get_named_compile_time_arg_val(moe_tail_named_args::kName)`). Plain constexpr only (no includes): the
// kernels include it as is.
//
// Which kernel takes which names:
//   latent engine, engine_dram_reader / engine_row_exchange_writer (engine_common.hpp): the engine group, the row
//     blocks; the reader also kRowsBlockStrideBytes, kSharedExpertBlockStrideBytes, kMaxShareTiles; the writer also
//     kOutputBlockStrideBytes
//   latent engine, engine_norm_projection_compute: kKHalfTiles, kEngineColumns, kMaxShareTiles,
//     kInverseLatentDimBits, kNormEpsBits, kExchangeTiles, kRowBlocks, kMaxSteps
//   latent engine, gather_port_staging_reader / gather_port_fabric_sender (gather_port_common.hpp): the gather port
//     group, kRowBlocks
//   all-to-all, the workers: kWorkerColumns, kHasSharedExpert (not the writer), kRowBlocks and the block strides of
//     the buffers the kernel addresses, kSkipPadding
//   all-to-all, port_staging_reader / port_fabric_sender (port_common.hpp): the all-to-all port group, kRowBlocks (the
//     sender also kOutputBlockStrideBytes), kSkipPadding
//   the all-to-all's dataflow kernels also read the padded forward's names of moe_head_named_args
//     (padded_forward_chunks, padded_forward_metadata: moe_head's padded_rows.hpp, shared by both ops)
//
// What is not a named arg: the TensorAccessorArgs chains (positional compile-time args from offset 0: each kernel's
// tensor accessors come first, in the order the host appends them).
#pragma once

namespace moe_tail_named_args {

// ---- row blocks (every kernel; each kernel loops over them itself) ----
inline constexpr const char* kRowBlocks = "row_blocks";
inline constexpr const char* kRowsBlockStrideBytes = "rows_block_stride_bytes";  // y_own's rows a block further, bytes
inline constexpr const char* kSharedExpertBlockStrideBytes = "shared_expert_block_stride_bytes";
inline constexpr const char* kOutputBlockStrideBytes = "output_block_stride_bytes";

// ---- a padded forward (padded_row_tiles.hpp; every all-to-all kernel) ----
inline constexpr const char* kSkipPadding = "skip_padding";  // 1: the padding row tiles are skipped

// ---- the latent engine (engine_common.hpp, the engine's kernels) ----
inline constexpr const char* kKHalfTiles = "k_half_tiles";               // the latent's K tiles a K half
inline constexpr const char* kRowTilesPerChunk = "row_tiles_per_chunk";  // a chip's own row tiles
inline constexpr const char* kNumChunks = "num_chunks";                  // the TP group's chips
inline constexpr const char* kEngineColumns = "engine_columns";
inline constexpr const char* kWTileBytes = "w_tile_bytes";  // W' is bfloat8_b or bfloat16
inline constexpr const char* kRingSlots = "ring_slots";
inline constexpr const char* kLatentTiles = "latent_tiles";
inline constexpr const char* kExchangeTiles = "exchange_tiles";
inline constexpr const char* kNumPairs = "num_pairs";
inline constexpr const char* kMaxSteps = "max_steps";
inline constexpr const char* kMaxShareTiles = "max_share_tiles";
inline constexpr const char* kInverseLatentDimBits = "inverse_latent_dim_bits";  // 1 / latent_dim, fp32 bits
inline constexpr const char* kNormEpsBits = "norm_eps_bits";                     // the rmsnorm's eps, fp32 bits

// ---- the latent engine's gather ports (gather_port_common.hpp) ----
inline constexpr const char* kTilesPerPacket = "tiles_per_packet";
inline constexpr const char* kPacketsPerGroup = "packets_per_group";
inline constexpr const char* kLandedGroupColumns = "landed_group_columns";
inline constexpr const char* kStagingCb = "staging_cb";  // (also the all-to-all ports')
inline constexpr const char* kPacketCountCb = "packet_count_cb";
inline constexpr const char* kPollCb = "poll_cb";
inline constexpr const char* kLandingCb = "landing_cb";

// ---- the all-to-all (the workers; port_common.hpp, the ports) ----
inline constexpr const char* kWorkerColumns = "worker_columns";
inline constexpr const char* kHasSharedExpert = "has_shared_expert";  // 1: the workers add the shared expert
inline constexpr const char* kPageBytes = "page_bytes";               // the ports' pages: bfloat16 tiles
inline constexpr const char* kPagesPerRun = "pages_per_run";
inline constexpr const char* kNumDramBanks = "num_dram_banks";
inline constexpr const char* kRunsPerGroup = "runs_per_group";
inline constexpr const char* kArrivalIncrementEvery = "arrival_increment_every";

}  // namespace moe_tail_named_args
