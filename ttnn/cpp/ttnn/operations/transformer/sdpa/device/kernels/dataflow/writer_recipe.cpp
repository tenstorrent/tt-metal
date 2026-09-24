// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "sequence_accessor.hpp"
#ifndef SDPA_RECIPE_DHT
#define SDPA_RECIPE_DHT 4
#endif
void kernel_main() {
    constexpr uint32_t q_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t primary_rows = get_compile_time_arg_val(1);
    constexpr uint32_t joint_rows = get_compile_time_arg_val(2);
    constexpr auto oa = TensorAccessorArgs<3>();
#ifdef SDPA_JOINT
    constexpr auto joa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto out = sequence_accessor<primary_rows, joint_rows, q_tiles * 32, SDPA_RECIPE_DHT>(
        TensorAccessor(oa, get_arg_val<uint32_t>(0)), TensorAccessor(joa, get_arg_val<uint32_t>(3)));
#else
    const auto out = sequence_accessor<primary_rows, q_tiles * 32, SDPA_RECIPE_DHT>(TensorAccessor(oa, get_arg_val<uint32_t>(0)));
#endif
    const uint32_t first_job = get_arg_val<uint32_t>(1);
    const uint32_t jobs = get_arg_val<uint32_t>(2);
    Noc noc;
    DataflowBuffer cb(16);
    for (uint32_t job = first_job; job < first_job + jobs; ++job) {
        for (uint32_t row = 0; row < q_tiles; ++row) {
            cb.wait_front(SDPA_RECIPE_DHT);
            for (uint32_t col = 0; col < SDPA_RECIPE_DHT; ++col) {
                out.visit(job * q_tiles * SDPA_RECIPE_DHT + row * SDPA_RECIPE_DHT + col, [&](const auto& destination, uint32_t page) {
                    noc.async_write(cb, destination, 2048, {.offset_bytes = col * 2048}, {.page_id = page});
                });
            }
            noc.async_write_barrier();
            cb.pop_front(SDPA_RECIPE_DHT);
        }
    }
}
