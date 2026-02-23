// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include "experimental/endpoints.h"

namespace generic = norm::kernel_util::generic;

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t NCHt = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    uint32_t tile_offset = get_arg_val<uint32_t>(3);

    uint32_t gamma_addr = get_arg_val<uint32_t>(5);
    uint32_t beta_addr = get_arg_val<uint32_t>(6);
    uint32_t b_addr = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_in0 = get_named_compile_time_arg_val("cb_in"),
                       cb_id_in1 = get_named_compile_time_arg_val("cb_inb");
    constexpr uint32_t cb_id_gamma = get_named_compile_time_arg_val("cb_gamma");
    constexpr uint32_t cb_id_beta = get_named_compile_time_arg_val("cb_beta");

    experimental::Noc noc;
    experimental::CircularBuffer cb_in0(cb_id_in0);
#ifdef FUSE_PRE_ADD
    experimental::CircularBuffer cb_in1(cb_id_in1);
#endif
#ifdef FUSE_GAMMA
    experimental::CircularBuffer cb_gamma(cb_id_gamma);
#endif
#ifdef FUSE_BETA
    experimental::CircularBuffer cb_beta(cb_id_beta);
#endif

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat src0_data_format = get_dataformat(cb_id_in0);

    constexpr uint32_t blk = get_compile_time_arg_val(0);  // needed for correctness of softmax/LN kernels
    constexpr bool use_welford = get_compile_time_arg_val(1) == 1;
    constexpr auto src0_args = TensorAccessorArgs<2>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<src1_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    constexpr uint32_t stick_size = get_compile_time_arg_val(beta_args.next_compile_time_args_offset());

    const auto src_a = TensorAccessor(src0_args, src_addr, src0_tile_bytes);

#ifdef FUSE_GAMMA
    const uint32_t gamma_tile_bytes = get_tile_size(cb_id_gamma);
    const auto addrg = TensorAccessor(gamma_args, gamma_addr, stick_size);
#endif
#ifdef FUSE_BETA
    const uint32_t beta_tile_bytes = get_tile_size(cb_id_beta);
    const auto addrb = TensorAccessor(beta_args, beta_addr, stick_size);
#endif
#ifdef FUSE_PRE_ADD
    const uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    const auto src_b = TensorAccessor(src1_args, b_addr, src1_tile_bytes);
#endif

    // Generate constant tiles for layernorm compute
    if constexpr (!use_welford) {
        constexpr uint32_t cb_in_2 = get_named_compile_time_arg_val("cb_scaler");
        dataflow_kernel_lib::
            calculate_and_prepare_reduce_scaler<cb_in_2, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    }
    constexpr uint32_t eps_cb_id = get_named_compile_time_arg_val("cb_eps");
    const uint32_t eps = get_arg_val<uint32_t>(4);
    generate_bcast_col_scalar(eps_cb_id, eps);

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t offs = 0;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        for (auto block : generic::blocks(Wt, blk)) {
            cb_in0.reserve_back(block.full_block_size());
            uint32_t idx = 0;
            for (auto r : block.local()) {
                noc.async_read(
                    src_a,
                    cb_in0,
                    src0_tile_bytes,
                    {.page_id = offs + block.start() + r + tile_offset},
                    {.offset_bytes = idx * src0_tile_bytes});
                idx++;
            }
            noc.async_read_barrier();
            cb_in0.push_back(block.full_block_size());

#ifdef FUSE_PRE_ADD
            cb_in1.reserve_back(block.full_block_size());
            idx = 0;
            for (auto r : block.local()) {
                noc.async_read(
                    src_b,
                    cb_in1,
                    src1_tile_bytes,
                    {.page_id = offs + block.start() + r + tile_offset},
                    {.offset_bytes = idx * src1_tile_bytes});
                idx++;
            }
            noc.async_read_barrier();
            cb_in1.push_back(block.full_block_size());
#endif
        }  // wt loop

#if defined FUSE_GAMMA || defined FUSE_BETA
        if (ncht == 0) {
            for (auto block : generic::blocks(Wt, blk)) {
#ifdef FUSE_GAMMA
                {
                    cb_gamma.reserve_back(block.full_block_size());
                    experimental::UnicastEndpoint local_ep;
                    uint32_t idx = 0;
                    for (auto r : block.local()) {
                        noc.async_read(
                            addrg,
                            cb_gamma,
                            64,
                            {.page_id = block.start() + r},
                            {.offset_bytes = idx * gamma_tile_bytes});
                        noc.async_read_barrier();
                        noc.async_read(
                            local_ep,
                            cb_gamma,
                            32,
                            {.noc_x = my_x[noc.get_noc_id()],
                             .noc_y = my_y[noc.get_noc_id()],
                             .addr = cb_gamma.get_write_ptr() + idx * gamma_tile_bytes + 32},
                            {.offset_bytes = idx * gamma_tile_bytes + 512});
                        idx++;
                    }
                    noc.async_read_barrier();
                    cb_gamma.push_back(block.full_block_size());
                }
#endif

#ifdef FUSE_BETA
                {
                    cb_beta.reserve_back(block.full_block_size());
                    experimental::UnicastEndpoint local_ep;
                    uint32_t idx = 0;
                    for (auto r : block.local()) {
                        noc.async_read(
                            addrb,
                            cb_beta,
                            64,
                            {.page_id = block.start() + r},
                            {.offset_bytes = idx * beta_tile_bytes});
                        noc.async_read_barrier();
                        noc.async_read(
                            local_ep,
                            cb_beta,
                            32,
                            {.noc_x = my_x[noc.get_noc_id()],
                             .noc_y = my_y[noc.get_noc_id()],
                             .addr = cb_beta.get_write_ptr() + idx * beta_tile_bytes + 32},
                            {.offset_bytes = idx * beta_tile_bytes + 512});
                        idx++;
                    }
                    noc.async_read_barrier();
                    cb_beta.push_back(block.full_block_size());
                }
#endif
            }  // wt loop
        }
#endif
        offs += Wt;
    }  // ncht loop
}
