// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/experimental/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/experimental/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

#include "debug/dprint.h"

template<uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);  // batch size
    constexpr uint32_t PNHt = get_compile_time_arg_val(1);  // padded number of heads in tiles
    constexpr uint32_t St = get_compile_time_arg_val(2);  // full sequence length of kv cache in tiles
    constexpr uint32_t DHt = get_compile_time_arg_val(3);  // head dim
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(4);
    constexpr uint32_t scale_val = get_compile_time_arg_val(5);
    constexpr uint32_t num_cores_per_batch = get_compile_time_arg_val(6);  // num cores per batch
    constexpr uint32_t num_cores = get_compile_time_arg_val(7);  // num running cores in total
    constexpr uint32_t in0_sender_semaphore_addr  = get_compile_time_arg_val(8);  // semaphore for sender
    constexpr uint32_t reduce_core_noc_x          = get_compile_time_arg_val(9);
    constexpr uint32_t reduce_core_noc_y          = get_compile_time_arg_val(10);
    constexpr uint32_t cur_batch = get_compile_time_arg_val(11);
    constexpr uint32_t worker_id = get_compile_time_arg_val(12);  // worker id among num_cores_per_batch-1 workers

    const uint32_t k_chunk_start = get_arg_val<uint32_t>(0);
    const uint32_t k_chunk_end = get_arg_val<uint32_t>(1);

    if (k_chunk_start == k_chunk_end) {
        // DPRINT << "[Writer Worker] No computes to be done for this worker" << ENDL();
        return; // early exit because no computes needs to be done for this worker
    }
    // DPRINT << "[Writer Worker] worker id, noc x and nox y: " << worker_id << ", " << reduce_core_noc_x << " " << reduce_core_noc_y << ENDL();

    const uint64_t in0_sender_semaphore_noc_addr = get_noc_addr(reduce_core_noc_x, reduce_core_noc_y, in0_sender_semaphore_addr);

    constexpr uint32_t out_chunk_tiles = PNHt * DHt;

    constexpr uint32_t cb_out = tt::CB::c_out0;
    constexpr uint32_t cb_out_m = tt::CB::c_out1;
    constexpr uint32_t cb_out_l = tt::CB::c_out2;
    constexpr uint32_t cb_intermed_out = tt::CB::c_out3;

    constexpr uint32_t cb_scale_in = tt::CB::c_in4;
    constexpr uint32_t cb_identity_scale_in = tt::CB::c_in5;

    generate_bcast_unary_scalar(cb_scale_in, scale_val);
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);

    // DPRINT << "[Writer Worker] Pushed statistics to copmute" << ENDL();

    uint32_t out_tile_id = 0;

    // Wait for compute to deliver output chunk
    cb_wait_front(cb_out, out_chunk_tiles);
    cb_wait_front(cb_out_m, PNHt);
    cb_wait_front(cb_out_l, PNHt);

    // DPRINT << "[Writer Worker] Received output chunk from compute" << ENDL();

    // Write output chunk to reducer
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t worker_offset = worker_id * (out_chunk_tiles+2*PNHt)*tile_bytes;
    constexpr uint32_t o_write_size = out_chunk_tiles*tile_bytes;
    constexpr uint32_t ml_write_size = PNHt*tile_bytes;
    uint64_t output_write_addr = get_noc_addr(reduce_core_noc_x, reduce_core_noc_y, get_write_ptr(cb_intermed_out)) + worker_offset;
    noc_async_write(get_read_ptr(cb_out), output_write_addr, o_write_size);
    output_write_addr+=o_write_size;
    noc_async_write(get_read_ptr(cb_out_m), output_write_addr, ml_write_size);
    output_write_addr+=ml_write_size;
    noc_async_write(get_read_ptr(cb_out_l), output_write_addr, ml_write_size);

    // DPRINT << "[Writer Worker] Wrote output chunk to reducer" << ENDL();

    // increment semaphore
    noc_async_write_barrier();
    noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);

    // DPRINT << "[Writer Worker] Incremented semaphore" << ENDL();

    // pop front
    cb_pop_front(cb_out, out_chunk_tiles);
    cb_pop_front(cb_out_m, PNHt);
    cb_pop_front(cb_out_l, PNHt);

    // DPRINT << "[Writer Worker] Done" << ENDL();
}
