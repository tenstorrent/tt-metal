// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the latent engine's engine-core data-movement kernels (engine_dram_reader, engine_row_exchange_writer):
// their common compile-time args 0 - 10 (moe_tail_program_factory engine_workload, from TailEngineGeometry; then
// each kernel's tensor accessors at ENGINE_ACCESSOR_ARGS and its row block args), the CB and semaphore ids, and the
// per-step words' layout.
//
// The engine: engine_columns x (2 num_pairs) cores; core (column, row) is in pair row % num_pairs and K half
// row / num_pairs. The call's row tiles are steps, dealt round-robin over the pairs (step s of pair p is the call's row
// tile s num_pairs + p). Row blocks (the kernels' own loop): a step's global index g = block x steps + s (steps: my
// pair's steps a block, the same on every core of a row and of its partner). The slots other cores address absolutely
// (the ring CB_RING and the squares slots CB_ROW_SQUARES by g % ring_slots, the exchange CB_EXCHANGE_RECEIVE by
// g % max_steps) follow the CBs' pointers, which advance a slot a step across blocks; the landing slots CB_LANDING are
// a block's (step s at slot s): the compute pushes a whole CB a block (its own steps' tiles, then filler) and the
// writer pops it whole, so the pointer is back at the base every block.

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"

// the latent's K tiles a K half (latent_dim / 32 / 2): every core's matmul runs over one half
constexpr uint32_t k_half_tiles = get_compile_time_arg_val(0);
// (compile-time arg 1: the most output tiles a column has - the compute's DEST bound, not read here)
// a chunk's row tiles (rows / 32 / tp_size): a chip's own rows, the unit the gather moves
constexpr uint32_t row_tiles_per_chunk = get_compile_time_arg_val(2);
constexpr uint32_t num_chunks = get_compile_time_arg_val(3);  // (= the TP group's chips)
constexpr uint32_t engine_columns = get_compile_time_arg_val(4);
constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(5);  // (W' is bfloat8_b or bfloat16)
constexpr uint32_t ring_slots = get_compile_time_arg_val(6);    // (chosen to fit the free L1)
constexpr uint32_t latent_tiles = get_compile_time_arg_val(7);  // (latent_dim / 32)
// a step's exchange with the partner half: its finished columns' partials (at most exchange_tiles - 1) and the half's
// sum of squares, fp32 tiles
constexpr uint32_t exchange_tiles = get_compile_time_arg_val(8);
constexpr uint32_t num_pairs = get_compile_time_arg_val(9);
constexpr uint32_t max_steps = get_compile_time_arg_val(10);  // (the most steps a pair has a block)
constexpr uint32_t ENGINE_ACCESSOR_ARGS = 11;

// Semaphore ids (moe_tail_program_factory: engine cores 0 .. 6, port cores 0 .. 1 - see gather_port_common.hpp):
//  PARTNER_EXCHANGE_SEM: the partner's exchanges landed here (cumulative over the blocks: wait g + 1);
//  SLOTS_FREE_SEM: the row's cores' steps done, their ring / squares slots free (cumulative:
//    engine_columns (g - ring_slots + 1));
//  W_LOADED_SEM: the W' K ranges landed, a bit a range (once a call);
//  W_GATE_COUNT_SEM / W_GATE_GO_SEM: the W' loads' gate (every core's first own rows read before any W' load);
//  BLOCK_GATE_COUNT_SEM / BLOCK_GATE_GO_SEM: the gate a row block (the lead's count-ins, the writers' go). Loop 0 opens
//    once every engine core zeroed its step words; block b > 0 once every engine writer and every port reader is done
//    with block b - 1 - nothing reads block b - 1's landing slots any more: the writers pop them, the port writers send
//    their ready (the previous chip lands block b's shares only after it). The lead also releases the gather ports
//    (gather_port_common.hpp PORT_READER_GO_SEM / PORT_WRITER_GO_SEM).
constexpr uint32_t PARTNER_EXCHANGE_SEM = 0, SLOTS_FREE_SEM = 1, W_LOADED_SEM = 2, W_GATE_COUNT_SEM = 3,
                   W_GATE_GO_SEM = 4, BLOCK_GATE_COUNT_SEM = 5, BLOCK_GATE_GO_SEM = 6;
constexpr uint32_t PORT_READER_GO_SEM = 0, PORT_WRITER_GO_SEM = 1;

// CB ids (moe_tail_program_factory's engine_cb, the same names)
constexpr uint32_t CB_UP_WEIGHTS = 0, CB_RING = 1, CB_ROW_SQUARES = 2, CB_MY_SQUARES = 3, CB_ONES = 4,
                   CB_EXCHANGE_SEND = 6, CB_EXCHANGE_RECEIVE = 8, CB_SHARED_EXPERT = 9, CB_OUTPUT = 10, CB_LANDING = 11,
                   CB_OWN_ROWS = 12, CB_STEP_WORDS = 16;

// The per-step words: this call's (CB_STEP_WORDS, right after the landing CB on the engine and the port cores: one
// address everywhere), zeroed once a call (block 0) by every engine core's reader before the zero gate, then counted
// per block - block b's thresholds are l + 1 (landed, a column's share) and (l + 1) engine_columns (the row's squares).
// A word is 16 B apart from the next: the L1 alignment of a NoC write / atomic, so one word's update never touches
// another's. In words of 16 B, in order:
//   kind 0, [0, max_steps): step s landed here (my share, own or gathered);
//   kind 1, [max_steps, 2 max_steps): only kind 1 step 0 is used - the source word of the share-count multicast;
//   kind 2, [2 max_steps, 3 max_steps): step s's squares tiles of the row in (engine_columns of them a block);
//   kind 3 step 0, at 3 max_steps: the block gate go's source word (the lead's multicast reads it);
//   then [3 max_steps + 1, + max_steps x engine_columns): step s's share of column c in (word s engine_columns + c).
constexpr uint32_t STEP_WORD_BYTES = 16;
constexpr uint32_t STEP_WORDS_BYTES = STEP_WORD_BYTES * (3 * max_steps + 1 + max_steps * engine_columns);
// the word of `kind` for `step` (kind 3: step 0 only); the share-arrival words come after them
FORCE_INLINE uint32_t step_word_offset(uint32_t kind, uint32_t step) {
    return STEP_WORD_BYTES * (kind * max_steps + step);
}
FORCE_INLINE uint32_t share_word_offset(uint32_t step, uint32_t column) {
    return STEP_WORD_BYTES * (3 * max_steps + 1 + step * engine_columns + column);
}
constexpr uint32_t STEP_LANDED = 0, STEP_SHARES = 1, STEP_SQUARES = 2, STEP_BLOCK_GO = 3;

FORCE_INLINE volatile tt_l1_ptr uint32_t* semaphore(uint32_t id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
}
// `step` of `pair`: the call's row tile
FORCE_INLINE uint32_t step_row_tile(uint32_t step, uint32_t pair) { return step * num_pairs + pair; }
