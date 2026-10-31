// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_gather_softmax/device/kernels/attn_res_core_runs.hpp"

void kernel_main() {
    // compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    // A deferred residual write to settle into the stream. Where there is none the
    // accessor below aliases running_sum, so the offsets hold either way and nothing
    // reads through it.
    constexpr bool fuse_add = get_compile_time_arg_val(1) == 1;
    constexpr auto prefix_args = TensorAccessorArgs<2>();
    constexpr auto partial_args = TensorAccessorArgs<prefix_args.next_compile_time_args_offset()>();
    constexpr auto q_args = TensorAccessorArgs<partial_args.next_compile_time_args_offset()>();
    constexpr auto pending_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();

    using Split = AttnResSplit<pending_args.next_compile_time_args_offset()>;
    constexpr uint32_t chunk_tiles = Split::chunk_tiles;
    const auto [first_unit, num_units] = Split::core_units();

    // Everything else is common: the partial's read site, in whole Ht*Wt planes, and
    // the buffer addresses. A program-cache hit patches these
    // once per program instead of once per core, and dispatch multicasts them.
    const auto partial_page_offset = get_common_arg_val<uint32_t>(0);
    const auto prefix_addr = get_common_arg_val<uint32_t>(1);
    const auto partial_addr = get_common_arg_val<uint32_t>(2);
    const auto q_addr = get_common_arg_val<uint32_t>(3);
    const auto pending_addr = get_common_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_scaler = 1;
    constexpr uint32_t cb_id_q = 2;
    constexpr uint32_t cb_id_prefix = 3;
    constexpr uint32_t cb_id_partial = 0;
    constexpr uint32_t cb_id_pending = 10;

    constexpr uint32_t prefix_tile_bytes = get_tile_size(cb_id_prefix);
    constexpr uint32_t partial_tile_bytes = get_tile_size(cb_id_partial);
    constexpr uint32_t q_tile_bytes = get_tile_size(cb_id_q);

    Noc noc;
    DataflowBuffer prefix_buf(cb_id_prefix);
    DataflowBuffer partial_buf(cb_id_partial);
    DataflowBuffer q_buf(cb_id_q);
    DataflowBuffer pending_buf(cb_id_pending);

    auto prefix_accessor = TensorAccessor(prefix_args, prefix_addr);
    auto partial_accessor = TensorAccessor(partial_args, partial_addr);
    auto q_accessor = TensorAccessor(q_args, q_addr);
    auto pending_accessor = TensorAccessor(pending_args, pending_addr);

    // Both reductions are plain sums; the mean this feeds is taken against the full
    // unsharded `d` downstream, not against this rank's share.
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_id_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    // q spans the same d as a row of the stream. A core's units are all the same chunk of
    // their rows — one unit where rows are split, whole rows where they are not — so it
    // needs q's span of that chunk only, read once and left resident.
    const uint32_t q_first = Split::unit(first_unit).chunk * chunk_tiles;
    q_buf.reserve_back(chunk_tiles);
    for (uint32_t k = 0; k < chunk_tiles; ++k) {
        noc.async_read(q_accessor, q_buf, q_tile_bytes, {.page_id = q_first + k}, {.offset_bytes = k * q_tile_bytes});
    }
    noc.async_read_barrier();
    q_buf.push_back(chunk_tiles);

    // Pass one's operands, a unit at a time: the live stream, and the deferred write
    // settled into it. Compute keeps what it reduces resident for the fold.
    for (uint32_t u = first_unit; u < first_unit + num_units; ++u) {
        const uint32_t first_tile = Split::template first_tile<Wt>(u);
        prefix_buf.reserve_back(chunk_tiles);
        if constexpr (fuse_add) {
            pending_buf.reserve_back(chunk_tiles);
        }
        for (uint32_t k = 0; k < chunk_tiles; ++k) {
            noc.async_read(
                prefix_accessor,
                prefix_buf,
                prefix_tile_bytes,
                {.page_id = first_tile + k},
                {.offset_bytes = k * prefix_tile_bytes});
            if constexpr (fuse_add) {
                noc.async_read(
                    pending_accessor,
                    pending_buf,
                    prefix_tile_bytes,
                    {.page_id = first_tile + k},
                    {.offset_bytes = k * prefix_tile_bytes});
            }
        }
        noc.async_read_barrier();
        prefix_buf.push_back(chunk_tiles);
        if constexpr (fuse_add) {
            pending_buf.push_back(chunk_tiles);
        }
    }

    // The fold's other operand, the sealed partial, prefetched behind pass one: it is
    // what the exchange leaves this core waiting with.
    for (uint32_t u = first_unit; u < first_unit + num_units; ++u) {
        const uint32_t first_tile = Split::template first_tile<Wt>(u);
        partial_buf.reserve_back(chunk_tiles);
        for (uint32_t k = 0; k < chunk_tiles; ++k) {
            noc.async_read(
                partial_accessor,
                partial_buf,
                partial_tile_bytes,
                {.page_id = first_tile + k + partial_page_offset},
                {.offset_bytes = k * partial_tile_bytes});
        }
        noc.async_read_barrier();
        partial_buf.push_back(chunk_tiles);
    }
}
