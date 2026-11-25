// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The flat routed expert's compile-time options, as the kernels see them.
//
// The host (flat_routed_expert_program_factory.cpp, flat_option_args) gives every flat-expert kernel the same named
// compile-time args, one per option below; an option a configuration does not use is 0. The kernels read them here
// once, as constexpr values, and pick code paths with `if constexpr` - no preprocessor defines. A value only one kernel
// (or one kind of core) uses is that kernel's own named compile-time arg, read in the kernel (the kernel's header
// comment lists them). The only positional compile-time args are the DRAM tensors' accessor args (flat_accessors.hpp:
// a TensorAccessorArgs chain is offset-based), which every data-movement kernel carries from index 0.
#pragma once

#include <cstdint>

#include "api/compile_time_args.h"

namespace flat_options {

// ---- the dynamic schedule (expert_schedule.hpp), the same in every kernel so every core builds the same schedule ----
// local experts a schedule can hold (16, 32 or 64: sizes its arrays)
constexpr uint32_t max_experts = get_named_compile_time_arg_val("max_experts");
// token rows per sub-block (row tiles per sub-block x 32)
constexpr uint32_t rows_per_sub_block = get_named_compile_time_arg_val("rows_per_sub_block");
// gate/up weight ring regions: how many experts' gate/up weights the ring holds at once
constexpr uint32_t gate_up_ring_regions = get_named_compile_time_arg_val("gate_up_ring_regions");
// pinning (expert_schedule.hpp, schedule_pin_hot_expert): the hot expert's chunk size in sub-blocks; 0: no pinning
constexpr uint32_t pin_chunk_sub_blocks = get_named_compile_time_arg_val("pin_chunk_sub_blocks");
// pin only when the hot expert has at least this many times the other experts' mean sub-blocks; 0: always
constexpr uint32_t pin_min_ratio = get_named_compile_time_arg_val("pin_min_ratio");
// with pinning, the down weights are read once per gate/up load into that load's ring region (as the gate/up weights)
// instead of once per schedule entry
constexpr bool down_weights_per_load = get_named_compile_time_arg_val("down_weights_per_load") != 0;
// bytes of CB 7 scratch a RISC reads the schedule's DRAM rows into (each RISC owns two of them)
constexpr uint32_t schedule_scratch_bytes = get_named_compile_time_arg_val("schedule_scratch_bytes");
// bytes of the schedule page a data-movement kernel hands its compute kernel (CB 6, schedule_meta.hpp)
constexpr uint32_t schedule_page_bytes = get_named_compile_time_arg_val("schedule_page_bytes");

// ---- the problem's shape, the same in every kernel ----
// local experts of a chip (the schedule's max_experts is the capacity above it)
constexpr uint32_t local_experts = get_named_compile_time_arg_val("local_experts");
// row tiles of a sub-block (the token rows of a sub-block, in tiles: rows_per_sub_block / 32)
constexpr uint32_t row_tiles_per_sub_block = rows_per_sub_block / 32;
// hidden size and intermediate size, in tiles
constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
constexpr uint32_t intermediate_tiles = get_named_compile_time_arg_val("intermediate_tiles");
// bytes of an x tile (after the tilize) and of an h tile (after gate/up)
constexpr uint32_t x_tile_bytes = get_named_compile_time_arg_val("x_tile_bytes");
constexpr uint32_t h_tile_bytes = get_named_compile_time_arg_val("h_tile_bytes");

// ---- x ----
// K tiles of an x super-block (a row-major chunk of x is super_block_tiles x 32 columns)
constexpr uint32_t super_block_tiles = get_named_compile_time_arg_val("super_block_tiles");
// x is read through a token index (x row token_index[r] for flat row r) instead of row by row
constexpr bool x_indexed = get_named_compile_time_arg_val("x_indexed") != 0;
// super-blocks of a sub-block (a sub-block's K tiles are that many super_block_tiles)
constexpr uint32_t super_blocks_per_sub_block = get_named_compile_time_arg_val("super_blocks_per_sub_block");
// K tiles of an x block (a gate/up weight block's, too): the x blocks of a sub-block are hidden_tiles / x_block_k_tiles
constexpr uint32_t x_block_k_tiles = get_named_compile_time_arg_val("x_block_k_tiles");
// x pieces a gate/up core's x block arrives in (the x arrival semaphore counts pieces)
constexpr uint32_t x_chain_pieces = get_named_compile_time_arg_val("x_chain_pieces");
// semaphore: the pieces of x that arrived in a gate/up core's x ring (on the head from the primary relay, else from the
// predecessor); and the count of x blocks the core's successor freed (the forwarder's, on the predecessor)
constexpr uint32_t x_arrived_sem = get_named_compile_time_arg_val("x_arrived_sem");
constexpr uint32_t x_slot_free_sem = get_named_compile_time_arg_val("x_slot_free_sem");
// x primary relays: the semaphore on the helpers that this primary returns freed landing slots on
constexpr uint32_t x_credit_sem = get_named_compile_time_arg_val("x_credit_sem");
// gate/up cores: the semaphores the x receiver publishes its consumed-block count and its end count in, for the x
// forwarder on the other RISC
constexpr uint32_t x_consumed_sem = get_named_compile_time_arg_val("x_consumed_sem");
constexpr uint32_t x_end_sem = get_named_compile_time_arg_val("x_end_sem");

// ---- gate/up weights (gate_up_weight_read.hpp) ----
// tiles of a gate/up weight block, and the weight ring's blocks (0: one load, gate_up_blocks_per_expert)
constexpr uint32_t gate_up_block_tiles = get_named_compile_time_arg_val("gate_up_block_tiles");
constexpr uint32_t gate_up_weight_ring_blocks = get_named_compile_time_arg_val("gate_up_weight_ring_blocks");
constexpr uint32_t gate_up_block_bytes = get_named_compile_time_arg_val("gate_up_block_bytes");
constexpr uint32_t gate_up_blocks_per_expert = get_named_compile_time_arg_val("gate_up_blocks_per_expert");
constexpr uint32_t gate_up_reads_in_flight = get_named_compile_time_arg_val("gate_up_reads_in_flight");
constexpr uint32_t gate_up_read_pieces = get_named_compile_time_arg_val("gate_up_read_pieces");
// the receiver hands the weight reader its loads list through this semaphore (a mailbox word)
constexpr uint32_t gate_up_mailbox_sem = get_named_compile_time_arg_val("gate_up_mailbox_sem");
// the receiver's count of landing slots granted to the weight reads, and the reads' count of landed blocks
constexpr uint32_t gate_up_grant_sem = get_named_compile_time_arg_val("gate_up_grant_sem");
constexpr uint32_t gate_up_data_sem = get_named_compile_time_arg_val("gate_up_data_sem");

// ---- gate/up compute and the h exchange ----
// gate/up cores (every one writes a slice of h to the down chain heads), and the gate/up tile pairs of a core
constexpr uint32_t gate_up_core_count = get_named_compile_time_arg_val("gate_up_core_count");
constexpr uint32_t gate_up_tile_pairs = get_named_compile_time_arg_val("gate_up_tile_pairs");
// the h buffers of a down core (h(v) lives in buffer v % h_buffers) and the semaphores counting the gate/up slices that
// arrived in buffers 0, 1 and 2 (one shared count could be completed by a fast core's next slice)
constexpr uint32_t h_buffers = get_named_compile_time_arg_val("h_buffers");
constexpr uint32_t h_gather_sem_0 = get_named_compile_time_arg_val("h_gather_sem_0");
constexpr uint32_t h_gather_sem_1 = get_named_compile_time_arg_val("h_gather_sem_1");
constexpr uint32_t h_gather_sem_2 = get_named_compile_time_arg_val("h_gather_sem_2");
// a gate/up core's go semaphore (the down cores' coordinator releases the next h with it)
constexpr uint32_t go_sem = get_named_compile_time_arg_val("go_sem");
// gate/up weights interleaved at 16 columns (half tiles) instead of 32: the activation works on faces (glu_faces.hpp)
constexpr bool half_width_gate_up = get_named_compile_time_arg_val("half_width_gate_up") != 0;
// 16-column units a gate/up core computes per row (0: whole gate/up tile pairs, 2 x NP tiles a row)
constexpr uint32_t gate_up_units = get_named_compile_time_arg_val("gate_up_units");
// h tiles a gate/up core produces per row with 16-column units
constexpr uint32_t gate_up_units_h_tiles = get_named_compile_time_arg_val("gate_up_units_h_tiles");
// row passes: row tiles per pass when DEST holds fewer than a whole sub-block's gate/up tiles (0: one pass)
constexpr uint32_t row_pass_rows = get_named_compile_time_arg_val("row_pass_rows");
// x blocks the gate/up core's x ring holds (row passes re-read x from it)
constexpr uint32_t x_ring_blocks = get_named_compile_time_arg_val("x_ring_blocks");

// ---- down and y ----
// down cores
constexpr uint32_t down_core_count = get_named_compile_time_arg_val("down_core_count");
// semaphores of a down core: h pieces that arrived from the predecessor, the h freed (reported to the predecessor), and
// the coordinator's count of the done reports
constexpr uint32_t h_arrived_sem = get_named_compile_time_arg_val("h_arrived_sem");
constexpr uint32_t h_slot_free_sem = get_named_compile_time_arg_val("h_slot_free_sem");
constexpr uint32_t down_done_sem = get_named_compile_time_arg_val("down_done_sem");
// y tile columns of a down core, K tiles of a down weight block and the block's tiles, blocks an expert's down weights
// take: the values of the down core's column width (one kernel set a width; 0 on the other kernels)
constexpr uint32_t down_columns = get_named_compile_time_arg_val("down_columns");
constexpr uint32_t down_k_block_tiles = get_named_compile_time_arg_val("down_k_block_tiles");
constexpr uint32_t down_block_tiles = get_named_compile_time_arg_val("down_block_tiles");
// (the divisor is never a literal 0, even in the branch the 0 skips: GCC's -Wdiv-by-zero sees through the ternary)
constexpr uint32_t down_blocks_per_expert =
    down_k_block_tiles ? intermediate_tiles / (down_k_block_tiles ? down_k_block_tiles : 1u) : 0;
// y is written row major (bf16 rows), not as bfp8 tiles: the down compute pack-untilizes, and down_weight_reader
// (NCRISC, NoC 0) writes the rows (y_row_major.hpp) - not down_core_writer (BRISC), whose NoC 1 carries the h chain
constexpr bool y_row_major = get_named_compile_time_arg_val("y_row_major") != 0;
// row tiles of y in flight per down core (the row-major out CB's capacity)
constexpr uint32_t y_row_major_rows = get_named_compile_time_arg_val("y_row_major_rows");
// DRAM banks the down weight blocks rotate over (block k of a core sits in bank (first + k) % banks)
constexpr uint32_t down_weight_banks = get_named_compile_time_arg_val("down_weight_banks");
// the down weight ring's capacity in blocks
constexpr uint32_t down_weight_ring_blocks = get_named_compile_time_arg_val("down_weight_ring_blocks");

// ---- running together with the combine (routed_experts_combine) ----
// The y writers' (down_weight_reader) per-core runtime-arg indices of their combine args, 0 on other kernels and when
// the op does not run with the combine: the writer index, the progress reports to the combine (combine_done.hpp),
// the L1 hand-off of y rows (y_row_major.hpp; 0 when the combine takes y from DRAM)
constexpr uint32_t combine_writer_index_arg = get_named_compile_time_arg_val("combine_writer_index_arg");
constexpr uint32_t combine_report_arg = get_named_compile_time_arg_val("combine_report_arg");
constexpr uint32_t combine_handoff_arg = get_named_compile_time_arg_val("combine_handoff_arg");
constexpr bool reports_to_combine = combine_report_arg != 0;
constexpr bool hands_y_to_combine = combine_handoff_arg != 0;

}  // namespace flat_options
