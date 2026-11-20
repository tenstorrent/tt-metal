// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/persistent_counter.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/relay_chunk_scan/device/kernels/dataflow/gate_matmul_dataflow.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/relay_chunk_scan/device/kernels/dataflow/relay_chronology.hpp"

// The chronology (relay_chronology.hpp), from common arg 0.
using Chronology = kda_relay::Chronology<0, 0>;
constexpr uint32_t correction_ct = Chronology::next_ct;
constexpr uint32_t kt = get_compile_time_arg_val(correction_ct);  // key tiles; the state is kt x kt tiles
// The entry correction: the seeds its chunks start from (the entry state, and P for a separated tail), and the chunk
// terms F(0) and F(I), interleaved and streamed a chunk row at a time, or in place in the core's shard.
constexpr uint32_t cb_seed = get_compile_time_arg_val(correction_ct + 1);
constexpr uint32_t cb_tail = get_compile_time_arg_val(correction_ct + 2);
constexpr uint32_t cb_terms = get_compile_time_arg_val(correction_ct + 3);
constexpr uint32_t core_chunks = get_compile_time_arg_val(correction_ct + 4);
constexpr bool streamed = get_compile_time_arg_val(correction_ct + 5);
constexpr uint32_t landed_semaphore_id = get_compile_time_arg_val(correction_ct + 6);
// The cores are heads x cores_per_head consecutive cores of the compute grid in row-major order, from first_core,
// head by head.
constexpr uint32_t cores_per_head = get_compile_time_arg_val(correction_ct + 7);
constexpr uint32_t grid_x = get_compile_time_arg_val(correction_ct + 8);
constexpr uint32_t first_core = get_compile_time_arg_val(correction_ct + 9);
constexpr auto terms_args = TensorAccessorArgs<correction_ct + 10>();
// The gated output norm: the mean's scaler, each chunk's gate rows (g of the head's chunks, formed by the matmul on
// the core) and the weight.
constexpr uint32_t norm_ct = terms_args.next_compile_time_args_offset();
constexpr uint32_t cb_scaler = get_compile_time_arg_val(norm_ct);
constexpr uint32_t cb_gate = get_compile_time_arg_val(norm_ct + 1);
constexpr uint32_t cb_weight = get_compile_time_arg_val(norm_ct + 2);
constexpr uint32_t heads = get_compile_time_arg_val(norm_ct + 3);
constexpr auto weight_args = TensorAccessorArgs<norm_ct + 4>();
// The output gate's operands (gate_matmul_dataflow.hpp): this kernel moves the head's W, on NoC 0.
using Gate = kda_gate::Operands<weight_args.next_compile_time_args_offset()>;
constexpr uint32_t state_tiles = kt * kt;
constexpr uint32_t num_chunks = Chronology::local_rows / tt::constants::TILE_HEIGHT;
// The common args after the chronology's: the terms, the norm weight, then every correction core's NoC x | y << 16.
constexpr uint32_t terms_arg = Chronology::next_common;
constexpr uint32_t weight_arg = terms_arg + 1;
constexpr uint32_t cores_arg = weight_arg + 1;

// Every correction core of a head corrects its share of the head's chunks from the seeds its relay core writes into
// it, and moves the head's W for its own matmul and the head's other cores (the writer moves x).
void kernel_main() {
    // Every core takes the same args (common, per device): its head and share follow from its place in the grid.
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x() - first_core;
    const uint32_t head = core / cores_per_head;
    const uint32_t place = core % cores_per_head;
    const uint32_t first_chunk = place * core_chunks;

    const Chronology chronology;
    // This core's valid chunks; those before the reset start from the entry state, the rest from the tail's seed.
    const uint32_t valid = chronology.interval.valid_chunks(0, 1);
    const uint32_t reset = chronology.interval.reset_chunk(0, 1);
    const uint32_t end_chunk = first_chunk + core_chunks < valid ? first_chunk + core_chunks : valid;
    const uint32_t chunks = end_chunk > first_chunk ? end_chunk - first_chunk : 0;
    uint32_t group_chunks = chunks;
    if (reset != 0) {
        group_chunks = reset <= first_chunk ? 0 : (reset - first_chunk < chunks ? reset - first_chunk : chunks);
    }
    // The writer's chronology: this core's chunk count; compute's: its chunk count and how many start from the entry.
    auto* chron_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(Chronology::cb_chron));
    auto* ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(Chronology::cb_ctl));
    chron_words[0] = chunks;
    cb_push_back(Chronology::cb_chron, 1);
    ctl[0] = chunks;
    ctl[1] = group_chunks;
    cb_push_back(Chronology::cb_ctl, 1);

    // The head's W for the core's matmul, then the partial sums with the core at this place of the pair's other head.
    Gate::move_w(head);
    if constexpr (Gate::ways == 2) {
        Gate::swap(get_common_arg_val<uint32_t>(cores_arg + Gate::partner(head) * cores_per_head + place));
    }

    // The norm's scaler and weight.
    if (chunks != 0) {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_scaler,
            ckernel::PoolType::AVG,
            ckernel::ReduceDim::REDUCE_ROW,
            kt * tt::constants::TILE_WIDTH>();
        const auto weight = TensorAccessor(weight_args, get_common_arg_val<uint32_t>(weight_arg));
        const uint32_t tile = get_tile_size(cb_weight);
        cb_reserve_back(cb_weight, kt);
        for (uint32_t j = 0; j < kt; ++j) {
            noc_async_read_page(j, weight, get_write_ptr(cb_weight) + j * tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_weight, kt);
    }

    // The chunks that start from the entry state. Their terms are in place, or streamed a chunk row at a time: head h's
    // F(0) rows are tile rows h * 2 * num_chunks + chunk of the terms, its F(I) rows num_chunks further.
    const auto stream_terms = [&](uint32_t begin, uint32_t end) {
        const auto terms = TensorAccessor(terms_args, get_common_arg_val<uint32_t>(terms_arg));
        const uint32_t tile = get_tile_size(cb_terms);
        for (uint32_t c = begin; c < end; ++c) {
            const uint32_t zero_page = (head * 2 * num_chunks + first_chunk + c) * kt;
            cb_reserve_back(cb_terms, 2 * kt);
            for (uint32_t j = 0; j < kt; ++j) {
                noc_async_read_page(zero_page + j, terms, get_write_ptr(cb_terms) + j * tile);
                noc_async_read_page(zero_page + num_chunks * kt + j, terms, get_write_ptr(cb_terms) + (kt + j) * tile);
            }
            noc_async_read_barrier();
            cb_push_back(cb_terms, 2 * kt);
        }
    };
    if (!streamed && chunks != 0) {
        cb_reserve_back(cb_terms, 2 * core_chunks * kt);
        cb_push_back(cb_terms, 2 * core_chunks * kt);
    }
    // A seed's tile columns (column-major by tile), as its relay core writes them here: the entry state's are counts
    // 1..kt of the landed semaphore (a program semaphore, zero at launch), the tail's kt + 1..2 kt.
    const uint32_t landed = get_semaphore(landed_semaphore_id);
    const auto receive = [&](uint32_t cb, uint32_t counted) {
        cb_reserve_back(cb, state_tiles);
        for (uint32_t j = 0; j < kt; ++j) {
            kda_counter::wait(landed, counted + j + 1);
            cb_push_back(cb, kt);
        }
    };
    if (group_chunks != 0) {
        receive(cb_seed, 0);
    }
    if constexpr (streamed) {
        stream_terms(0, group_chunks);
    }
    // The separated tail's chunks start from P.
    if (chunks != group_chunks) {
        receive(cb_tail, kt);
    }
    if constexpr (streamed) {
        stream_terms(group_chunks, chunks);
    }
    noc_async_atomic_barrier();
}
