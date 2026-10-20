// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "sequence_accessor.hpp"

// Named CTAs (host: run_recipe_segments, sdpa_recipe.cpp).
namespace recipe_writer {
// Joint SDPA: the joint output's accessor follows the primary one, its address is runtime arg 3.
constexpr bool has_joint = get_named_compile_time_arg_val("joint") != 0;
constexpr uint32_t d_tiles = get_named_compile_time_arg_val("d_tiles");
// Zero Q rows padding a paired recipe's odd chunk (host: recipe_compute_q_tiles).
constexpr uint32_t q_pad_tiles = get_named_compile_time_arg_val("q_pad_tiles");
}  // namespace recipe_writer
using namespace recipe_writer;

// The output as one sequence: the primary output and, in joint builds, the joint output (args at joint_args).
template <uint32_t q_tiles, uint32_t primary_rows, uint32_t joint_rows, uint32_t args_offset>
FORCE_INLINE auto recipe_output_sequence() {
    constexpr auto oa = TensorAccessorArgs<args_offset>();
    if constexpr (has_joint) {
        constexpr auto joa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
        return sequence_accessor<primary_rows, joint_rows, q_tiles * 32, d_tiles>(
            TensorAccessor(oa, get_arg_val<uint32_t>(0)), TensorAccessor(joa, get_arg_val<uint32_t>(3)));
    } else {
        return sequence_accessor<primary_rows, q_tiles * 32, d_tiles>(TensorAccessor(oa, get_arg_val<uint32_t>(0)));
    }
}

void kernel_main() {
    constexpr uint32_t q_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t primary_rows = get_compile_time_arg_val(1);
    constexpr uint32_t joint_rows = get_compile_time_arg_val(2);
    const auto out = recipe_output_sequence<q_tiles, primary_rows, joint_rows, 3>();
    const uint32_t first_job = get_arg_val<uint32_t>(1);
    const uint32_t jobs = get_arg_val<uint32_t>(2);
    Noc noc;
    DataflowBuffer cb(16);
    for (uint32_t job = first_job; job < first_job + jobs; ++job) {
        for (uint32_t row = 0; row < q_tiles; ++row) {
            cb.wait_front(d_tiles);
            for (uint32_t col = 0; col < d_tiles; ++col) {
                out.visit(job * q_tiles * d_tiles + row * d_tiles + col, [&](const auto& destination, uint32_t page) {
                    noc.async_write(cb, destination, 2048, {.offset_bytes = col * 2048}, {.page_id = page});
                });
            }
            noc.async_write_barrier();
            cb.pop_front(d_tiles);
        }
        // Drop the padding rows of a paired recipe's odd chunk (host: recipe_compute_q_tiles).
        for (uint32_t row = 0; row < q_pad_tiles; ++row) {
            cb.wait_front(d_tiles);
            cb.pop_front(d_tiles);
        }
    }
}
