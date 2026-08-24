// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/groupnorm_constants.hpp"
#include "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/groupnorm_zero_fill.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

// split REDUCE across cores
void kernel_main() {
    constexpr uint32_t num_mcast_cores = get_compile_time_arg_val(0);
    constexpr uint32_t num_batch_group = get_compile_time_arg_val(1);

    constexpr uint32_t per_core_N = get_compile_time_arg_val(2);
    const uint32_t per_core_N_bytes = get_compile_time_arg_val(3);
    const uint32_t per_core_N_bytes_with_stride = get_compile_time_arg_val(4);
    constexpr uint32_t datum_size_bytes = get_compile_time_arg_val(5);
    // Per-core slots in dfb_ex_external are hardcoded to a dfb_ex_external_slot_pitch_bytes
    // pitch (see the `l1_write_addr_external += dfb_ex_external_slot_pitch_bytes`
    // increments below). Each NOC read writes datum_size_bytes into its slot, so
    // datum_size_bytes > dfb_ex_external_slot_pitch_bytes would overflow into the
    // next core's slot and silently corrupt the reduction. The slot pitch itself
    // would need to grow to support larger datums.
    static_assert(
        datum_size_bytes <= dfb_ex_external_slot_pitch_bytes,
        "dfb_ex_external slot pitch is hardcoded; "
        "datum_size_bytes must be <= dfb_ex_external_slot_pitch_bytes or per-slot writes will overflow");
    constexpr uint32_t per_core_M = get_compile_time_arg_val(6);
    constexpr uint32_t tile_height = get_compile_time_arg_val(7);

    tt_l1_ptr uint32_t* noc_coord_x = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(0));
    tt_l1_ptr uint32_t* noc_coord_y = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(num_mcast_cores));

    constexpr uint32_t operation_rt_args_end = 2 * num_mcast_cores;
    constexpr dataflow_kernel_lib::McastArgs<8, operation_rt_args_end> reduction_mcast_args;

    Noc noc;
    Semaphore<> reduce_receiver_sem(reduction_mcast_args.consumer_ready);
    auto reduction_pipe = reduction_mcast_args.sender(noc);

    constexpr uint32_t dfb_ex_partial_id = tt::CBIndex::c_8;
    constexpr uint32_t dfb_ex_id = tt::CBIndex::c_9;
    constexpr uint32_t dfb_ex_external_id = tt::CBIndex::c_10;
    constexpr uint32_t dfb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t dfb_repack_id = tt::CBIndex::c_11;
    constexpr uint32_t dfb_repack_out_id = tt::CBIndex::c_12;
    constexpr uint32_t dfb_out0_id = tt::CBIndex::c_16;

    DataflowBuffer dfb_ex_partial(dfb_ex_partial_id);
    DataflowBuffer dfb_ex(dfb_ex_id);
    DataflowBuffer dfb_ex_external(dfb_ex_external_id);
    DataflowBuffer dfb_in0(dfb_in0_id);
    DataflowBuffer dfb_repack(dfb_repack_id);
    DataflowBuffer dfb_repack_out(dfb_repack_out_id);
    DataflowBuffer dfb_out0(dfb_out0_id);

    const uint32_t single_tile_size_bytes = get_tile_size(dfb_ex_partial_id);
    const DataFormat data_format = get_dataformat(dfb_ex_partial_id);
    const uint32_t num_bytes_read = datum_size_bytes;

#if defined(READER_REPACK) and defined(TILIZE_IN)
    uint32_t in0_l1_read_addr = dfb_in0.get_read_ptr();
    uint32_t src_addr_in0 = in0_l1_read_addr;
    UnicastEndpoint self_ep;
    for (uint32_t m = 0; m < per_core_M; ++m) {
        dfb_repack.reserve_back(per_core_N);
        uint32_t l1_write_addr_repack = dfb_repack.get_write_ptr();
        for (uint32_t i = 0; i < tile_height; ++i) {
            noc.async_read(
                self_ep,
                CoreLocalMem<uint32_t>(l1_write_addr_repack),
                per_core_N_bytes,
                {.noc_x = my_x[0], .noc_y = my_y[0], .addr = src_addr_in0},
                {});
            src_addr_in0 += per_core_N_bytes;
            l1_write_addr_repack += per_core_N_bytes_with_stride;
        }
        noc.async_read_barrier();
        dfb_repack.push_back(per_core_N);
    }
#endif

    // fp32 breaks the full-tile self-read's REDUCE_SCALAR packer-zero contract, so zero dfb_ex_external up front and
    // read each scalar at datum width; bf16 keeps the cheaper full-tile trick.
    constexpr bool stats_fp32_zero_fill = (datum_size_bytes >= 4);
    if constexpr (stats_fp32_zero_fill) {
        zero_whole_cb(dfb_ex_external_id, noc);
    }

    if constexpr (num_mcast_cores > 1) {
        for (uint32_t m = 0; m < num_batch_group; ++m) {
            for (uint32_t n = 0; n < 2; ++n) {
                dfb_ex_partial.wait_front(1);

                uint32_t l1_read_addr_ex_par = dfb_ex_partial.get_read_ptr();
                dfb_ex_external.reserve_back(1);
                uint32_t l1_write_addr_external = dfb_ex_external.get_write_ptr();

                // Self read uses single_tile_size_bytes (not num_bytes_read) on
                // purpose: it doubles as a free zero-init of every byte in the
                // reserved tile other than this core's own slot.
                // The producer of dfb_ex_partial (compute/groupnorm_sharded_v2.cpp)
                // pushes a tile produced by `reduce<PoolType::SUM, ReduceDim::REDUCE_SCALAR>`,
                // and the LLK packer for REDUCE_SCALAR is documented to write the
                // scalar result at face-0 [0, 0] and explicitly clear every other
                // datum in the tile via its edge masks.
                //
                // Therefore, after this read, dfb_ex_external's reserved tile contains:
                //   - bytes [0, datum_size_bytes): local core's scalar (slot 0).
                //   - bytes [datum_size_bytes, single_tile_size_bytes): exact zero.
                // The remote-core reads below then overwrite slot bytes
                // [dfb_ex_external_slot_pitch_bytes*i,
                //  dfb_ex_external_slot_pitch_bytes*i + datum_size_bytes) for
                // i = 1 .. num_mcast_cores-1.
                // All gap bytes, per-slot bytes
                // [datum_size_bytes, dfb_ex_external_slot_pitch_bytes) and any
                // trailing-tile bytes past slot num_mcast_cores-1, stay zero, so
                // the downstream reduce_tile sum on dfb_ex_external is not
                // polluted.
                UnicastEndpoint remote_ep;
                // fp32: read self at datum width (gaps zeroed up front); bf16: full-tile self-read zero-inits the
                // reserved tile.
                noc.async_read(
                    remote_ep,
                    CoreLocalMem<uint32_t>(l1_write_addr_external),
                    stats_fp32_zero_fill ? num_bytes_read : single_tile_size_bytes,
                    {.noc_x = noc_coord_x[0], .noc_y = noc_coord_y[0], .addr = l1_read_addr_ex_par},
                    {});
                l1_write_addr_external += dfb_ex_external_slot_pitch_bytes;
                noc.async_read_barrier();

                // The acknowledgement counter protects the following remote source-L1 reads.
                // The multicast helper is intentionally no-handshake for this phase.
                reduce_receiver_sem.wait(num_mcast_cores - 1);
                reduce_receiver_sem.set(0);
                for (uint32_t i = 0; i < num_mcast_cores - 1; ++i) {
                    UnicastEndpoint remote_ep;
                    noc.async_read(
                        remote_ep,
                        CoreLocalMem<uint32_t>(l1_write_addr_external),
                        num_bytes_read,
                        {.noc_x = noc_coord_x[i + 1], .noc_y = noc_coord_y[i + 1], .addr = l1_read_addr_ex_par},
                        {});
                    l1_write_addr_external += dfb_ex_external_slot_pitch_bytes;
                    noc.async_read_barrier();
                }
                dfb_ex_external.push_back(1);

                dfb_ex.wait_front(1);
                dfb_ex_partial.pop_front(1);

                uint32_t l1_read_addr_ex = dfb_ex.get_read_ptr();
                reduction_pipe.send(l1_read_addr_ex, l1_read_addr_ex, num_bytes_read);
                dfb_ex.pop_front(1);
            }
        }
    }

#if defined(READER_REPACK) and defined(UNTILIZE_OUT)
    uint32_t l1_write_addr_repack = dfb_out0.get_write_ptr();
    for (uint32_t m = 0; m < per_core_M; ++m) {
        dfb_repack_out.wait_front(per_core_N);
        uint32_t in0_l1_read_addr = dfb_repack_out.get_read_ptr();
        uint32_t src_addr_in0 = in0_l1_read_addr;
        UnicastEndpoint self_ep;
        for (uint32_t i = 0; i < tile_height; ++i) {
            noc.async_read(
                self_ep,
                CoreLocalMem<uint32_t>(l1_write_addr_repack),
                per_core_N_bytes,
                {.noc_x = my_x[0], .noc_y = my_y[0], .addr = src_addr_in0},
                {});
            src_addr_in0 += per_core_N_bytes_with_stride;
            l1_write_addr_repack += per_core_N_bytes;
        }
        noc.async_read_barrier();
        dfb_repack_out.pop_front(per_core_N);
    }
#endif
}
