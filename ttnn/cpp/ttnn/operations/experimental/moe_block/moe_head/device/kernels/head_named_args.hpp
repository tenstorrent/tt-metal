// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE head kernels' named compile-time args: one name for each meaning, shared by the host (head_*_program.cpp:
// `named_arg(kernel, moe_head_named_args::kName, value)`) and the kernels (`get_named_compile_time_arg_val(
// moe_head_named_args::kName)`). Plain constexpr only (no includes): the kernels include it as is.
//
// A kernel reads a name only where it needs it, but the host always gives every kernel that includes a header the
// names the header reads (0, or the named off value, when the feature is off) - an `if constexpr` branch not taken
// still compiles, so its names must exist.
//
// What is not a named arg:
//   * the TensorAccessorArgs chains (positional compile-time args, from the offsets named below) and the
//     kernel_lib McastArgs (their `<prefix>_ct_offset` / `<prefix>_rt_offset` names come from the library);
//   * the per-chip values (ROUTING_EXCHANGE_MY_CHIP, ROUTING_PLANNER_MY_CHIP / _PORTS / _NUM_PORTS, ROUTER_MY_CHIP,
//     ROUTER_TP_POSITION, WATCHER_ROUTING_ARGS_OFFSET): defines whose value is a read of the chip's block of the DRAM
//     table (head_program.cpp chip_values), so that the mesh runs one program - a compile-time value would give every
//     chip its own;
//   * the list-valued ROUTE_WORDS_*_INIT defines (router_common.hpp); ROUTER_SQRTSOFTPLUS (it selects the SFPU headers
//     router_compute includes) with the LLK's INP_FLOAT32; the tt-metal build macros.
#pragma once

namespace moe_head_named_args {

// ---- row blocks (row_block.hpp; every kernel) ----
inline constexpr const char* kRowBlocks = "row_blocks";  // the call's row blocks, a kernel_main loop's count
inline constexpr const char* kRowBlockBarrierArg = "row_block_barrier_arg";    // the barrier table's first common arg
inline constexpr const char* kRowBlockBarrierLead = "row_block_barrier_lead";  // 1: this kernel leads its cores
inline constexpr const char* kRowBlockBarrierCoordinator = "row_block_barrier_coordinator";  // 1: the coordinator
inline constexpr const char* kRowBlockXStrideBytes = "row_block_x_stride_bytes";
inline constexpr const char* kRowBlockYStrideBytes = "row_block_y_stride_bytes";
inline constexpr const char* kRowBlockDispatchRowStride = "row_block_dispatch_row_stride";
inline constexpr const char* kRowBlockPaddedRowStride = "row_block_padded_row_stride";
inline constexpr const char* kRowBlockRouterPageStride = "row_block_router_page_stride";
inline constexpr const char* kRowBlockRouteWordsStride = "row_block_route_words_stride";
inline constexpr const char* kRowBlockStrides[] = {
    kRowBlockXStrideBytes,
    kRowBlockYStrideBytes,
    kRowBlockDispatchRowStride,
    kRowBlockPaddedRowStride,
    kRowBlockRouterPageStride,
    kRowBlockRouteWordsStride};

// ---- a padded forward (padded_rows.hpp, x_gather.hpp; the gather's ports and watcher, the engine, the router
// kernels) ----
inline constexpr const char* kSkipPaddingChunks = "skip_padding_chunks";           // 1: run the live row chunks only
inline constexpr const char* kSkipPaddingArgsOffset = "skip_padding_args_offset";  // the live args' first common arg
inline constexpr const char* kLiveChunksFlagId = "live_chunks_flag_id";            // the ewords flag of the live chunks
inline constexpr const char* kPaddedChunk = "padded_chunk";  // 1: the router kernels read actual_start / actual_end
inline constexpr const char* kPaddedForwardChunks = "padded_forward_chunks";  // chunks packed in the forward (1: one)
inline constexpr const char* kPaddedForwardMetadata = "padded_forward_metadata";  // 1: the forward's metadata words

// ---- the x gather's ports (x_gather.hpp, latent_all_to_all.hpp, reduce_scatter.hpp) ----
inline constexpr const char* kKStepTiles = "k_step_tiles";
inline constexpr const char* kRowTilesPerCore = "row_tiles_per_core";
inline constexpr const char* kKStepsPerShard = "k_steps_per_shard";
inline constexpr const char* kTilesPerPacket = "tiles_per_packet";
inline constexpr const char* kChunkRowTiles = "chunk_row_tiles";
inline constexpr const char* kRowChunks = "row_chunks";
inline constexpr const char* kGatherLinks = "gather_links";
inline constexpr const char* kPacketsPerGroup = "packets_per_group";
inline constexpr const char* kPacketsPerArrivalIncrement = "packets_per_arrival_increment";
inline constexpr const char* kEngineColumns = "engine_columns";
inline constexpr const char* kEngineRows = "engine_rows";
inline constexpr const char* kFamily0Columns = "family_0_columns";
inline constexpr const char* kFamily1Columns = "family_1_columns";
inline constexpr const char* kTpSize = "tp_size";
inline constexpr const char* kXGatherHandoffBytes = "x_gather_handoff_bytes";
inline constexpr const char* kLatentChunkBytes = "latent_chunk_bytes";
inline constexpr const char* kReduceScatterPasses = "reduce_scatter_passes";
inline constexpr const char* kRouterLogitsFirst = "router_logits_first";  // 1: the logits go ahead of the x gather

// ---- the routing exchange and planner (routing_planner.hpp, router_fabric_sends.hpp) ----
inline constexpr const char* kRoutingExchangeRouterCores = "routing_exchange_router_cores";
inline constexpr const char* kRoutingExchangeRouterRowTiles = "routing_exchange_router_row_tiles";
inline constexpr const char* kRoutingExchangeNumChips = "routing_exchange_num_chips";
inline constexpr const char* kRoutingExchangePlannerInstances = "routing_exchange_planner_instances";
inline constexpr const char* kRoutingExchangeExpertsPerChip = "routing_exchange_experts_per_chip";
inline constexpr const char* kRoutingExchangeHeaderBytes = "routing_exchange_header_bytes";
inline constexpr const char* kRoutingExchangeExpertCountBlockBytes = "routing_exchange_expert_count_block_bytes";
inline constexpr const char* kRoutingExchangeInstanceBlockBytes = "routing_exchange_instance_block_bytes";
inline constexpr const char* kRoutingExchangeMaxPacketPayloadBytes = "routing_exchange_max_packet_payload_bytes";
inline constexpr const char* kRoutingAccessorsOffset = "routing_accessors_compile_time_args_offset";
inline constexpr const char* kRoutingPlannerAccessorsOffset = "routing_planner_accessors_compile_time_args_offset";
inline constexpr const char* kRoutingPlannerTokens = "routing_planner_tokens";
inline constexpr const char* kTopK = "top_k";
inline constexpr const char* kSpRows = "sp_rows";
inline constexpr const char* kTpColumns = "tp_columns";
inline constexpr const char* kRoutingPlannerAccumulators = "routing_planner_accumulators";
inline constexpr const char* kRoutingPlannerLinks = "routing_planner_links";
inline constexpr const char* kRoutingPlannerTokensPerPage = "routing_planner_tokens_per_page";
inline constexpr const char* kRoutingPlannerSlotBytes = "routing_planner_slot_bytes";
inline constexpr const char* kRoutingPlannerRowsCapacity = "routing_planner_rows_capacity";
inline constexpr const char* kRoutingPlannerEntriesCapacity = "routing_planner_entries_capacity";
inline constexpr const char* kRoutingPlannerIdle = "routing_planner_idle";  // 1: an earlier row block's own program
// (1: both planners, the block picks)
inline constexpr const char* kRoutingPlannerPerBlock = "routing_planner_per_block";
inline constexpr const char* kRoutingPlannerIdleTokens = "routing_planner_idle_tokens";
inline constexpr const char* kRoutingPlannerIdleRowsCapacity = "routing_planner_idle_rows_capacity";
inline constexpr const char* kRoutingPlannerIdleEntriesCapacity = "routing_planner_idle_entries_capacity";
inline constexpr const char* kRoutingPlannerNone = "routing_planner_none";  // 1: no instance on this kernel's cores

// ---- the router cores (router_common.hpp, router_input_feeder, router_compute, router_output_writer) ----
inline constexpr const char* kExpertTiles = "expert_tiles";
inline constexpr const char* kRouterFold = "router_fold";
inline constexpr const char* kRouterMathSyncOffset = "router_math_sync_offset";
inline constexpr const char* kRouterIterations = "router_iterations";  // the router phase's iterations (1: one)
inline constexpr const char* kRouterRowTilesPerIteration = "router_row_tiles_per_iteration";
inline constexpr const char* kRouterGoFlagId = "router_go_flag_id";           // router core 0's go word in the ewords
inline constexpr const char* kRouterIterationArgs = "router_iteration_args";  // the writer's iteration args' count
inline constexpr const char* kRouterScoreStride = "router_score_stride";
inline constexpr const char* kRouterRoutePageBytes = "router_route_page_bytes";
inline constexpr const char* kRouterRouteAccessorOffset = "router_route_accessor_compile_time_args_offset";
inline constexpr const char* kRouteWordsPerToken = "route_words_per_token";
inline constexpr const char* kRouteWordsExperts = "route_words_experts";
inline constexpr const char* kRouteWordsWrapX = "route_words_wrap_x";
inline constexpr const char* kRouteWordsWrapY = "route_words_wrap_y";

// ---- the shared expert's matmul cores (shared_expert.hpp, the engine's three kernels) ----
inline constexpr const char* kGateUpKSteps = "gate_up_k_steps";
inline constexpr const char* kGateUpPasses = "gate_up_passes";
inline constexpr const char* kMaxColumnPairs = "max_column_pairs";
inline constexpr const char* kXReadySemaphoreId = "x_ready_semaphore_id";
inline constexpr const char* kDownKStepTiles = "down_k_step_tiles";
inline constexpr const char* kDownKSteps = "down_k_steps";
inline constexpr const char* kDownPasses = "down_passes";
inline constexpr const char* kWSlots = "w_slots";
inline constexpr const char* kWSlotTiles = "w_slot_tiles";
inline constexpr const char* kWTileBytes = "w_tile_bytes";
inline constexpr const char* kWValidSemaphoreId = "w_valid_semaphore_id";
inline constexpr const char* kWFreeSemaphoreId = "w_free_semaphore_id";
inline constexpr const char* kKShards = "k_shards";
inline constexpr const char* kSharedExpertHSlots = "shared_expert_h_slots";
inline constexpr const char* kSharedExpertAccumulatorPages = "shared_expert_accumulator_pages";
inline constexpr const char* kSharedExpertDestTiles = "shared_expert_dest_tiles";
// (0: chunk-major)
inline constexpr const char* kSharedExpertOwnerChunkRowTiles = "shared_expert_owner_chunk_row_tiles";
inline constexpr const char* kSharedExpertOwnerRowTiles = "shared_expert_owner_row_tiles";
inline constexpr const char* kSharedExpertChunkRowTiles = "shared_expert_chunk_row_tiles";
inline constexpr const char* kSharedExpertWRingSlots = "shared_expert_w_ring_slots";  // 0: no W ring
inline constexpr const char* kSharedExpertGateUpPasses = "shared_expert_gate_up_passes";
// (0: no passes)
inline constexpr const char* kSharedExpertReduceScatterPasses = "shared_expert_reduce_scatter_passes";
// (1: the dispatch prep runs after the last chunk)
inline constexpr const char* kDispatchPrepLast = "dispatch_prep_last";
inline constexpr const char* kReduceScatterEntries = "reduce_scatter_entries";
inline constexpr const char* kReduceScatterHoldCb = "reduce_scatter_hold_cb";
inline constexpr const char* kReduceScatterFinalBlockCb = "reduce_scatter_final_block_cb";
inline constexpr const char* kReduceScatterArrivalForwardCb = "reduce_scatter_arrival_forward_cb";
inline constexpr const char* kReduceScatterArrivalBackwardCb = "reduce_scatter_arrival_backward_cb";
inline constexpr const char* kReduceScatterFinalCounterCb = "reduce_scatter_final_counter_cb";
inline constexpr const char* kReduceScatterOutputForwardCb = "reduce_scatter_output_forward_cb";
inline constexpr const char* kReduceScatterPassSyncCb = "reduce_scatter_pass_sync_cb";
inline constexpr const char* kReduceScatterFinalFreeCb = "reduce_scatter_final_free_cb";
inline constexpr const char* kReduceScatterFinalRows = "reduce_scatter_final_rows";  // 1: the final rows' form
inline constexpr const char* kReduceScatterZoneTilesForward = "reduce_scatter_zone_tiles_forward";
inline constexpr const char* kReduceScatterZoneTilesBackward = "reduce_scatter_zone_tiles_backward";
inline constexpr const char* kLatentEnabled = "latent_enabled";  // 1: the latent's all-to-all
inline constexpr const char* kLatentCb = "latent_cb";
inline constexpr const char* kLatentReadyCb = "latent_ready_cb";
inline constexpr const char* kLatentMaxColumnTiles = "latent_max_column_tiles";
inline constexpr const char* kLatentPages = "latent_pages";
inline constexpr const char* kRouterXCb = "router_x_cb";
inline constexpr const char* kRouterWCb = "router_w_cb";
inline constexpr const char* kRouterOutCb = "router_out_cb";
inline constexpr const char* kRouterKTiles = "router_k_tiles";
inline constexpr const char* kRouterExpertTilesPerColumn = "router_expert_tiles_per_column";
inline constexpr const char* kRouterCoreRowTiles = "router_core_row_tiles";
inline constexpr const char* kRouterRowTilesPerGroup = "router_row_tiles_per_group";
inline constexpr const char* kRouterDepth = "router_depth";
inline constexpr const char* kRouterKChunkTiles = "router_k_chunk_tiles";
inline constexpr const char* kRouterDoneCredit = "router_done_credit";  // 1: the router overlay's done credit
inline constexpr const char* kRouterAccessorOffset = "router_accessor_compile_time_args_offset";
inline constexpr const char* kXAcknowledgeFlagId = "x_acknowledge_flag_id";
inline constexpr const char* kWAcknowledgeFlagId = "w_acknowledge_flag_id";
inline constexpr const char* kRouterXReadyFlagId = "router_x_ready_flag_id";
inline constexpr const char* kRouterWReadyFlagId = "router_w_ready_flag_id";
inline constexpr const char* kRouterXSentFlagId = "router_x_sent_flag_id";
inline constexpr const char* kXReadLocalFlagId = "x_read_local_flag_id";
inline constexpr const char* kDispatchPrepUnitCb = "dispatch_prep_unit_cb";
inline constexpr const char* kDispatchPrepOutCb = "dispatch_prep_out_cb";
inline constexpr const char* kDispatchPrepFullColumnTiles = "dispatch_prep_full_column_tiles";
inline constexpr const char* kDispatchPrepBlockColumnTiles = "dispatch_prep_block_column_tiles";
inline constexpr const char* kDispatchPrepRowBytes = "dispatch_prep_row_bytes";
inline constexpr const char* kDispatchPrepAccessorOffset = "dispatch_prep_accessor_compile_time_args_offset";

}  // namespace moe_head_named_args
