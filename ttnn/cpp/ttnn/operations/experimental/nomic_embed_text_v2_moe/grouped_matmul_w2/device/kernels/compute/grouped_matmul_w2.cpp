// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Every compute core: for each expert in the schedule's order, takes the expert's weight slice once and multiplies
// this core's blocks of that expert's region by it. A block's output is one 1 x n_tiles subblock: all K steps
// accumulate in DST in k order with the weight tiles transposed on unpack, then the subblock is packed row-major
// once.
// A slice is two K halves in two rings, so a ring may hold more first halves than second ones; matmul_block
// addresses tiles from a ring's read pointer without wrapping, so each ring holds whole halves. The first block of an
// expert waits for each half piece by piece; the last pops each piece once its K steps are issued, so the next slice
// can stream into it. A block's input arrives in K pieces of input_piece_tiles, each waited for and popped on its
// own. Only the first and last blocks of an expert carry those weight checks: on every K step they cost 5% a block.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/operations/experimental/nomic_embed_text_v2_moe/grouped_matmul_w1/device/kernels/dataflow/expert_schedule.hpp"

namespace {

constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
constexpr uint32_t k_tiles = get_named_compile_time_arg_val("k_tiles");
constexpr uint32_t n_tiles = get_named_compile_time_arg_val("n_tiles");
constexpr uint32_t input_cb_id = get_named_compile_time_arg_val("input_cb");
constexpr uint32_t first_half_cb_id = get_named_compile_time_arg_val("first_half_cb");
constexpr uint32_t second_half_cb_id = get_named_compile_time_arg_val("second_half_cb");
constexpr uint32_t output_cb_id = get_named_compile_time_arg_val("output_cb");
constexpr uint32_t schedule_cb_id = get_named_compile_time_arg_val("schedule_cb");
constexpr uint32_t piece_tiles = get_named_compile_time_arg_val("piece_tiles");
constexpr uint32_t input_piece_tiles = get_named_compile_time_arg_val("input_piece_tiles");
constexpr uint32_t half_k = k_tiles / 2;
constexpr uint32_t half_tiles = half_k * n_tiles;
constexpr bool transpose = true;
static_assert(k_tiles % 2 == 0 && half_tiles % piece_tiles == 0 && half_k % input_piece_tiles == 0);

// One K half's ring: tiles of the current slice waited for and popped so far.
struct Half {
    uint32_t waited = 0;
    uint32_t released = 0;
};

// K steps [k_begin, k_end) of one half, the input piece's first tile at k_begin, waiting for each weight row (wait)
// and popping each weight piece past it (release).
template <uint32_t ring_cb_id, bool wait, bool release>
FORCE_INLINE void k_steps(uint32_t k_begin, uint32_t k_end, uint32_t half_begin, Half& half) {
    DataflowBuffer ring(ring_cb_id);
    for (uint32_t k = k_begin; k < k_end; ++k) {
        const uint32_t index = (k - half_begin) * n_tiles;
        if constexpr (wait) {
            if (index + n_tiles > half.waited) {
                half.waited = index + n_tiles;
                ring.wait_front(half.waited - half.released);
            }
        }
        matmul_block(input_cb_id, ring_cb_id, k - k_begin, index - half.released, 0, transpose, n_tiles, 1, k_tiles);
        if constexpr (release) {
            while (half.released + piece_tiles <= index + n_tiles) {
                ring.pop_front(piece_tiles);
                half.released += piece_tiles;
            }
        }
    }
}

template <bool wait, bool release>
FORCE_INLINE void block_k_steps(DataflowBuffer& input_cb, Half (&halves)[2]) {
    for (uint32_t k = 0; k < k_tiles; k += input_piece_tiles) {
        input_cb.wait_front(input_piece_tiles);
        if (k < half_k) {
            k_steps<first_half_cb_id, wait, release>(k, k + input_piece_tiles, 0, halves[0]);
        } else {
            k_steps<second_half_cb_id, wait, release>(k, k + input_piece_tiles, half_k, halves[1]);
        }
        input_cb.pop_front(input_piece_tiles);
    }
}

}  // namespace

void kernel_main() {
    DataflowBuffer input_cb(input_cb_id);
    DataflowBuffer output_cb(output_cb_id);
    DataflowBuffer schedule_cb(schedule_cb_id);
    DataflowBuffer rings[2] = {DataflowBuffer(first_half_cb_id), DataflowBuffer(second_half_cb_id)};

    compute_kernel_hw_startup<SrcOrder::Reverse>(input_cb_id, first_half_cb_id, output_cb_id);
    matmul_block_init(input_cb_id, first_half_cb_id, transpose, n_tiles, 1, k_tiles);
    pack_untilize_dest_init<n_tiles, n_tiles>(output_cb_id);

    schedule_cb.wait_front(1);
    for (uint32_t position = 0; position < num_experts; ++position) {
        const uint32_t expert_blocks = read_tile_value(schedule_cb_id, 0, position);
        if (expert_blocks == grouped_matmul::kSkipExpert) {
            continue;
        }
        Half halves[2];
        for (uint32_t block = 0; block < expert_blocks; ++block) {
            const bool first = block == 0;
            const bool last = block + 1 == expert_blocks;
            output_cb.reserve_back(n_tiles);
            tile_regs_acquire();
            if (first && last) {
                block_k_steps<true, true>(input_cb, halves);
            } else if (first) {
                block_k_steps<true, false>(input_cb, halves);
            } else if (last) {
                block_k_steps<false, true>(input_cb, halves);
            } else {
                block_k_steps<false, false>(input_cb, halves);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<n_tiles, n_tiles>(output_cb_id, 1, 0);
            tile_regs_release();
            output_cb.push_back(n_tiles);
        }
        // A lane with no block of the expert takes the slice and lets it go.
        for (uint32_t index = 0; index < 2; ++index) {
            Half& half = halves[index];
            if (half.released < half_tiles) {
                if (half.waited < half_tiles) {
                    rings[index].wait_front(half_tiles - half.released);
                }
                rings[index].pop_front(half_tiles - half.released);
            }
        }
    }
    schedule_cb.pop_front(1);
}
