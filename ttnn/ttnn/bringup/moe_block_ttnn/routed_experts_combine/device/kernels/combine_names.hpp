// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The combine kernels' named compile-time args: one name for each meaning, shared by the host (combine_program.cpp
// builds every kernel's named args from these) and the kernels (get_named_compile_time_arg_val(combine_names::k...)).
// Plain constexpr only (no includes): both sides include it as is.
//
// A kernel reads the names of its own role and the options; the roles' names are distinct where their values may
// differ, and one name has one value wherever several roles use it (a two-role kernel -
// combine_port_or_accumulator_reader / _writer.cpp - gets both roles' names merged, equal names with equal values).
// The only positional compile-time args are the TensorAccessorArgs chains (offset-based): each is named by the
// absolute index of its first arg (k...AccessorFirstArg). Every other value is named.
//
//   name                                   | value (offsets: bytes in a core's arena)
//   ---------------------------------------+---------------------------------------------------------------------------
//   (options)                              | combine_options.hpp: accumulators_on_ports, per_chip_accumulator_table,
//                                          |   shared_landing and its offsets, landing_on_flat_expert_cores, ...
//   injection_cb / port_table_cb           | CB of the port's injection ring / of the port reader's plan scratch
//   row_cb / meta_cb / weight_cb           | CBs the accumulator's reader fills for its compute: rows, meta pages, gate
//                                          |   weights
//   accumulator_table_cb                   | CB of the accumulator's plan scratch (CB_TAB)
//   accumulator_cb / output_cb             | CBs of the sums (int32 with the snap, else fp32) and of the bf16 rows
//   row_bytes / row_tiles                  | bytes of a bf16 row / its tiles (flat 2 KB)
//   landing_slot_bytes                     | bytes of a landing slot (a row + its 32 B trailer, 64 B aligned)
//   injection_slot_bytes                   | the same in the injection ring
//   landing_slots_per_accumulator          | landing pool slots at each accumulator (carved per call)
//   accumulator_count / chip_count         | accumulators of a chip / chips of the mesh
//   fabric_links                           | fabric links between neighbouring chips
//   max_tokens_per_accumulator             | tokens an accumulator holds at most
//   row_header_count / row_header_stride   | rotating row headers per destination chip / bytes between them
//   credits_received_offset, rows_sent_offset, credits_to_return_offset, misc_words_offset, row_headers_offset,
//   rings_offset, ring_layouts_received_offset, ring_layouts_to_forward_offset, rows_sent_at_call_start_offset:
//                                          | the state layout (combine_state_layout)
//   return_credits_after_rows              | rows consumed from a source before its credits go back
//   idle_snapshot_rows / _age_cycles       | idle: credits go back after that many rows, or when the last snapshot is
//                                          |   that many cycles old
//   barrier_call_slot                      | the RISC's call-count slot of the zero gate's barrier
//   y_writer_words_offset                  | the flat expert's y writers' words (arena-relative, on their cores)
//   accumulator_table_offset / _bytes      | the per-chip accumulator table's state offset / bytes
//   local_ring_offset                      | the accumulators' local ring (arena-relative)
//   local_ring_landed_word                 | misc word of the local ring's landed[] array
//   local_ring_handoff_base_word           | misc word of the local ring's hand-off base
//   snap_step_bits                         | fp32 bits of the snap step q: int32 sums of whole q's (0: off, fp32 sums)
//   port_role_first_runtime_arg            | two-role kernels: the runtime arg where the port role's args start
//   <role>_..._accessor_first_arg          | compile-time arg index of a TensorAccessorArgs chain (roles: port_reader,
//                                          |   accumulator_reader, port_writer, accumulator_writer)
#pragma once

namespace combine_names {

// ---- the combine's options (combine_options.hpp) ----
inline constexpr const char* kAccumulatorsOnPorts = "accumulators_on_ports";
inline constexpr const char* kPerChipAccumulatorTable = "per_chip_accumulator_table";
inline constexpr const char* kSharedLanding = "shared_landing";
inline constexpr const char* kSharedLandingGrantsPerSource = "shared_landing_grants_per_source";
inline constexpr const char* kSharedLandingGrantIdsOffset = "shared_landing_grant_ids_offset";
inline constexpr const char* kSharedLandingGrantSnapshotsOffset = "shared_landing_grant_snapshots_offset";
inline constexpr const char* kLandingOnFlatExpertCores = "landing_on_flat_expert_cores";
inline constexpr const char* kLandingReadsInFlight = "landing_reads_in_flight";

// ---- circular buffers ----
inline constexpr const char* kInjectionCb = "injection_cb";
inline constexpr const char* kPortTableCb = "port_table_cb";
inline constexpr const char* kRowCb = "row_cb";
inline constexpr const char* kMetaCb = "meta_cb";
inline constexpr const char* kWeightCb = "weight_cb";
inline constexpr const char* kAccumulatorTableCb = "accumulator_table_cb";
inline constexpr const char* kAccumulatorCb = "accumulator_cb";
inline constexpr const char* kOutputCb = "output_cb";

// ---- shapes ----
inline constexpr const char* kRowBytes = "row_bytes";
inline constexpr const char* kRowTiles = "row_tiles";
inline constexpr const char* kLandingSlotBytes = "landing_slot_bytes";
inline constexpr const char* kInjectionSlotBytes = "injection_slot_bytes";
inline constexpr const char* kLandingSlotsPerAccumulator = "landing_slots_per_accumulator";
inline constexpr const char* kAccumulatorCount = "accumulator_count";
inline constexpr const char* kChipCount = "chip_count";
inline constexpr const char* kFabricLinks = "fabric_links";
inline constexpr const char* kMaxTokensPerAccumulator = "max_tokens_per_accumulator";
inline constexpr const char* kRowHeaderCount = "row_header_count";
inline constexpr const char* kRowPieces = "row_pieces";
inline constexpr const char* kRowHeaderStride = "row_header_stride";

// ---- the state layout (combine_state_layout) and the credit policy ----
inline constexpr const char* kCreditsReceivedOffset = "credits_received_offset";
inline constexpr const char* kRowsSentOffset = "rows_sent_offset";
inline constexpr const char* kCreditsToReturnOffset = "credits_to_return_offset";
inline constexpr const char* kMiscWordsOffset = "misc_words_offset";
inline constexpr const char* kRowHeadersOffset = "row_headers_offset";
inline constexpr const char* kRingsOffset = "rings_offset";
inline constexpr const char* kRingLayoutsReceivedOffset = "ring_layouts_received_offset";
inline constexpr const char* kRingLayoutsToForwardOffset = "ring_layouts_to_forward_offset";
inline constexpr const char* kRowsSentAtCallStartOffset = "rows_sent_at_call_start_offset";
inline constexpr const char* kReturnCreditsAfterRows = "return_credits_after_rows";
inline constexpr const char* kIdleSnapshotRows = "idle_snapshot_rows";
inline constexpr const char* kIdleSnapshotAgeCycles = "idle_snapshot_age_cycles";

// ---- placement ----
inline constexpr const char* kBarrierCallSlot = "barrier_call_slot";
inline constexpr const char* kYWriterWordsOffset = "y_writer_words_offset";
inline constexpr const char* kAccumulatorTableOffset = "accumulator_table_offset";
inline constexpr const char* kAccumulatorTableBytes = "accumulator_table_bytes";
inline constexpr const char* kLocalRingOffset = "local_ring_offset";
inline constexpr const char* kLocalRingLandedWord = "local_ring_landed_word";
inline constexpr const char* kLocalRingHandoffBaseWord = "local_ring_handoff_base_word";
inline constexpr const char* kSnapStepBits = "snap_step_bits";
inline constexpr const char* kPortRoleFirstRuntimeArg = "port_role_first_runtime_arg";

// ---- TensorAccessorArgs chains: the absolute compile-time arg index of the first arg ----
inline constexpr const char* kPortReaderPlanAccessorFirstArg = "port_reader_plan_accessor_first_arg";
inline constexpr const char* kAccumulatorReaderPlanAccessorFirstArg = "accumulator_reader_plan_accessor_first_arg";
inline constexpr const char* kPortWriterOutputAccessorFirstArg = "port_writer_output_accessor_first_arg";
inline constexpr const char* kPortWriterPlanAccessorFirstArg = "port_writer_plan_accessor_first_arg";
inline constexpr const char* kPortWriterHeaderCacheAccessorFirstArg = "port_writer_header_cache_accessor_first_arg";
inline constexpr const char* kPortWriterAccumulatorTableAccessorFirstArg =
    "port_writer_accumulator_table_accessor_first_arg";
inline constexpr const char* kAccumulatorWriterOutputAccessorFirstArg = "accumulator_writer_output_accessor_first_arg";

}  // namespace combine_names
