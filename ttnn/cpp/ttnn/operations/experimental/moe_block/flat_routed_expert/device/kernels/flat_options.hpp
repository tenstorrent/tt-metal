// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The flat routed expert's compile-time options, as the kernels see them.
//
// The host (flat_routed_expert_program_factory.cpp, flat_option_args) gives every flat-expert kernel the same named
// compile-time args, one per option below; an option a configuration does not use is 0. The kernels read them here
// once, as constexpr values, and pick code paths with `if constexpr` - no preprocessor defines.
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
// data-movement kernels: the compile-time arg index where the DRAM tensors' accessor args start (flat_accessors.hpp)
constexpr uint32_t accessor_args_base = get_named_compile_time_arg_val("accessor_args_base");

// ---- x ----
// K tiles of an x super-block (a row-major chunk of x is super_block_tiles x 32 columns)
constexpr uint32_t super_block_tiles = get_named_compile_time_arg_val("super_block_tiles");
// x is read through a token index (x row token_index[r] for flat row r) instead of row by row
constexpr bool x_indexed = get_named_compile_time_arg_val("x_indexed") != 0;
// x pieces a gate/up core's x block arrives in (XARR counts pieces)
constexpr uint32_t x_chain_pieces = get_named_compile_time_arg_val("x_chain_pieces");
// x primary relays: the semaphore on the helpers that this primary returns freed landing slots on
constexpr uint32_t x_credit_sem = get_named_compile_time_arg_val("x_credit_sem");
// gate/up cores: the semaphores the x receiver publishes its consumed-block count and its end count in, for the x
// forwarder on the other RISC
constexpr uint32_t x_consumed_sem = get_named_compile_time_arg_val("x_consumed_sem");
constexpr uint32_t x_end_sem = get_named_compile_time_arg_val("x_end_sem");

// ---- gate/up weights (gate_up_weight_read.hpp) ----
constexpr uint32_t gate_up_block_bytes = get_named_compile_time_arg_val("gate_up_block_bytes");
constexpr uint32_t gate_up_blocks_per_expert = get_named_compile_time_arg_val("gate_up_blocks_per_expert");
constexpr uint32_t gate_up_reads_in_flight = get_named_compile_time_arg_val("gate_up_reads_in_flight");
constexpr uint32_t gate_up_read_pieces = get_named_compile_time_arg_val("gate_up_read_pieces");
// the receiver hands the weight reader its loads list through this semaphore (a mailbox word)
constexpr uint32_t gate_up_mailbox_sem = get_named_compile_time_arg_val("gate_up_mailbox_sem");
// the receiver's count of landing slots granted to the weight reads, and the reads' count of landed blocks
constexpr uint32_t gate_up_grant_sem = get_named_compile_time_arg_val("gate_up_grant_sem");
constexpr uint32_t gate_up_data_sem = get_named_compile_time_arg_val("gate_up_data_sem");

// ---- gate/up compute ----
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
