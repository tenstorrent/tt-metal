// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp
// (block-sharded 2D-mcast conv2d activation reader).
//
// Algorithm body identical to the legacy kernel; only the host-binding surface is migrated:
//   - CB-index CTAs -> dfb:: tokens (act / act_row_major / act_tilized / reader_indices / act_sharded)
//   - act-mcast semaphore-id CTAs -> sem::act_mcast_sender / sem::act_mcast_receiver
//   - remaining positional CTAs -> get_arg(args::name)
//   - RTAs -> get_arg(args::name); the per-core act-mcast sender NoC-coord lookup table (variable count =
//     number of cores in the mcast dimension) is supplied as positional runtime VARARGS, accessed via
//     get_vararg(i)
//   - DRAM config-tensor read uses tensor::reader_indices (CONFIG_TENSOR_IN_DRAM path)
//
// split_reader_cb_shared (the side-channel-semaphore second-writer overlap) is DEFERRED on the host side
// (the factory TT_FATAL-rejects it and forces split reader off), so all split_reader_cb_shared code paths
// and the reserve_done/write_done semaphores are removed here.  conv_reader_common.hpp helpers are
// templated on the CB-object type, so DataflowBuffers (constructed from dfb:: constexpr indices) are
// passed to them and to the local mcast helpers directly.

#include <cstdint>
#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "conv_reader_common.hpp"
#include "noc/noc_parameters.h"
#define ENABLE_DEBUG 0

#include "api/debug/dprint.h"  // [#47797] act-mcast handshake DPRINT trace (run with DPRINT on)
#if ENABLE_DEBUG
#include "api/debug/dprint_pages.h"
#endif

#include "api/debug/ring_buffer.h"
// DEBUG: block-sharded conv2d act-mcast deadlock localization via watcher ring buffer (remove after).
// Marker layout: 0xRP_IIII  R=role/kernel(A=act-reader), P=phase, IIII=(nbh<<8)|act_w_outer_i.
#define RB_ITER(nbh, awo) ((((uint32_t)(nbh)) << 8) | ((uint32_t)(awo) & 0xff))

// Multicasts activation data from src_cb to dst_cb across cores in the multicast rectangle.
// Three cases depending on the sender's role and number of multicast destinations:
//   is_receiver_core && act_mcast_num_cores > 0:  mcast with INCLUDE_SRC loopback
//   is_receiver_core && act_mcast_num_cores == 0: local self-write (mcast loopback hangs with 0 destinations)
//   !is_receiver_core:                            standard mcast EXCLUDE_SRC (even when act_mcast_num_cores == 0,
//                                                 because the sender still needs to send to the output core)
template <uint32_t act_mcast_num_cores>
void multicast_data(
    Noc noc,
    MulticastEndpoint mcast_ep,
    bool is_receiver_core,
    DataflowBuffer src_cb,
    uint32_t src_offset,
    McastDst& dst,
    uint32_t total_bytes) {
    auto& src = src_cb;  // DataflowBuffer src -> read_ptr (was use<READ_PTR> on CircularBuffer)
    if (is_receiver_core) {
        if constexpr (act_mcast_num_cores > 0) {
            noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                src, mcast_ep, total_bytes, act_mcast_num_cores + 1, {.offset_bytes = src_offset}, dst, true);
        } else {
            // Sender is the only receiver — can't use multicast loopback (hangs with 0 destinations)
            noc.async_write(
                src,
                UnicastEndpoint{},
                total_bytes,
                {.offset_bytes = src_offset},
                {.noc_x = my_x[noc.get_noc_id()], .noc_y = my_y[noc.get_noc_id()], .addr = dst.addr});
        }
    } else {
        noc.async_write_multicast(
            src, mcast_ep, total_bytes, act_mcast_num_cores + 1, {.offset_bytes = src_offset}, dst, true);
    }
}

// Multicast activation data from the local circular buffer to multiple destinations (dst_cb in receiver cores).
// This function sends a block of data (the activation block) using NOC multicast commands, it avoids waiting for the
// whole block to be available in the source CB before starting the multicast, instead waits for enough tiles to do one
// multicast of NOC_MAX_BURST_SIZE size. This is because under the hood, the multicast splits the data into chunks of
// NOC_MAX_BURST_SIZE size
// It calls the multicast_data function for each chunk of maximum size NOC_MAX_BURST_SIZE bytes.
// Said function does mcast loopback when the sender core is also a receiver core (it is both in output and input grids)
// or mcast when the sender core is not a receiver core (it is only present in the input grid, mcast loopback will hang
// if the core isn't one of receivers) or just local write when it is in both input and output grids but is the only
// receiver core (will hang if mcast loopback is used)
template <
    uint32_t act_mcast_num_dest_cores,
    uint32_t mcast_noc_burst_size,
    uint32_t block_tile_count,
    uint32_t tile_size>
void mcast_block_chunked(
    Noc noc,
    MulticastEndpoint mcast_ep,
    DataflowBuffer src_cb_obj,
    bool is_receiver_core,
    DataflowBuffer dst_cb_obj,
    const McastRect& rect) {
    // Build mcast dst once; only .addr is updated per burst
    // number of full bursts
    constexpr uint32_t mcast_full_burst_cnt = block_tile_count * tile_size / mcast_noc_burst_size;
    // size of the leftover burst, if 0 means we have no leftover burst
    constexpr uint32_t mcast_leftover_burst_size = block_tile_count * tile_size % mcast_noc_burst_size;
    // number of tiles that we need to wait for to cover the full burst size
    constexpr uint32_t wait_tile_full_cnt = (mcast_noc_burst_size + tile_size - 1) / tile_size;

    // In full burst iterations we wait for a bit more than the full burst size in case where the
    // tile size does not divide the burst size evenly.
    // we need to insure that we don't wait for more tiles than we have in the block
    constexpr uint32_t wait_tile_full_done = std::min(mcast_full_burst_cnt * wait_tile_full_cnt, block_tile_count);

    // optimization to avoid unnecessary branching in the loop
    constexpr bool no_need_partial_wait_tile = mcast_full_burst_cnt * wait_tile_full_cnt <= block_tile_count;

    // number of times we need to increase the wait_tile_curr for the full burst iterations
    constexpr uint32_t wait_tile_full_iter_cnt = (wait_tile_full_done / wait_tile_full_cnt) - 1;

    uint32_t src_offset = 0;
    McastDst dst = {
        .noc_x_start = rect.noc_x_start,
        .noc_y_start = rect.noc_y_start,
        .noc_x_end = rect.noc_x_end,
        .noc_y_end = rect.noc_y_end,
        .addr = dst_cb_obj.get_write_ptr()};

    constexpr uint32_t wait_tile_start_cnt = std::min(block_tile_count, wait_tile_full_cnt);
    uint32_t wait_tile_curr = wait_tile_start_cnt;
    for (uint32_t i = 0; i < mcast_full_burst_cnt; i++) {
        src_cb_obj.wait_front(wait_tile_curr);
        multicast_data<act_mcast_num_dest_cores>(
            noc, mcast_ep, is_receiver_core, src_cb_obj, src_offset, dst, mcast_noc_burst_size);
        src_offset += mcast_noc_burst_size;
        dst.addr += mcast_noc_burst_size;

        if constexpr (no_need_partial_wait_tile) {
            wait_tile_curr += wait_tile_full_cnt;
        } else {
            // we shouldn't wait for more than the number of tiles in the block
            if (i < wait_tile_full_iter_cnt) {
                wait_tile_curr += wait_tile_full_cnt;
            } else {
                wait_tile_curr = block_tile_count;
            }
        }
    }
    if constexpr (mcast_leftover_burst_size > 0) {
        src_cb_obj.wait_front(block_tile_count);
        multicast_data<act_mcast_num_dest_cores>(
            noc, mcast_ep, is_receiver_core, src_cb_obj, src_offset, dst, mcast_leftover_burst_size);
    }

    // In case we only do local l1 writes, we need to wait for the barrier to complete
    if constexpr (act_mcast_num_dest_cores == 0) {
        if (is_receiver_core) {
            noc.async_write_barrier();
        }
    }
}

void kernel_main() {
    constexpr uint32_t dilation_h = get_arg(args::dilation_h);
    constexpr uint32_t dilation_w = get_arg(args::dilation_w);
    constexpr uint32_t stride_w = get_arg(args::stride_w);
    constexpr uint32_t conv_act_c_read_bytes = get_arg(args::conv_act_c_read_bytes);
    constexpr uint32_t window_outer = get_arg(args::window_outer);
    constexpr uint32_t act_block_num_tiles_read = get_arg(args::act_block_num_tiles_read);
    constexpr uint32_t weight_size_h = get_arg(args::weight_size_h);
    constexpr uint32_t weight_size_w = get_arg(args::weight_size_w);
    constexpr uint32_t padded_conv_act_size_w = get_arg(args::padded_conv_act_size_w);
    constexpr uint32_t act_block_w_extra_align_bytes = get_arg(args::act_block_w_extra_align_bytes);
    constexpr uint32_t act_num_blocks_h = get_arg(args::act_num_blocks_h);
    constexpr uint32_t act_block_num_tiles = get_arg(args::act_block_num_tiles);
    constexpr uint32_t act_w_num_outer = get_arg(args::act_w_num_outer);
    constexpr uint32_t act_mcast_num_dests = get_arg(args::act_mcast_num_dests);
    constexpr uint32_t act_mcast_num_cores = get_arg(args::act_mcast_num_cores);
    constexpr uint32_t act_mcast_tile_size_bytes = get_arg(args::act_mcast_tile_size_bytes);
    constexpr bool transpose_mcast = get_arg(args::transpose_mcast) == 1;
    constexpr bool needs_act_block_zero_out = get_arg(args::needs_act_block_zero_out) == 1;
    constexpr uint32_t cb_id_act = dfb::act;
    constexpr uint32_t tilized_in0_cb_id = dfb::act_tilized;
    constexpr uint32_t cb_id_act_row_major_bfloat16 = dfb::act_row_major;
    constexpr bool split_reader_enabled = get_arg(args::split_reader_enabled);

    // Experimental API objects
    Noc noc;
    Semaphore act_mcast_sender_sem(sem::act_mcast_sender);
    Semaphore act_mcast_receiver_sem(sem::act_mcast_receiver);
    MulticastEndpoint mcast_ep;
    DataflowBuffer cb_act_obj(cb_id_act);
    DataflowBuffer cb_act_rm_obj(cb_id_act_row_major_bfloat16);
    DataflowBuffer cb_tilized_in0_obj(tilized_in0_cb_id);

    if constexpr (needs_act_block_zero_out) {
        zero_out_tiles<cb_id_act_row_major_bfloat16>(noc, cb_act_rm_obj);
    }

    const McastRect act_mcast_rect = {
        get_arg(args::mcast_dest_noc_start_x),
        get_arg(args::mcast_dest_noc_start_y),
        get_arg(args::mcast_dest_noc_end_x),
        get_arg(args::mcast_dest_noc_end_y)};
    uint32_t act_mcast_sender_id = get_arg(args::act_mcast_sender_id);
    uint32_t act_mcast_sender_noc_x = get_arg(args::act_mcast_sender_noc_x);
    const bool is_receiver_core = get_arg(args::is_receiver_core) > 0;
    const bool is_sender_core = get_arg(args::is_sender_core) > 0;
    uint32_t dram_config_reader_index = get_arg(args::dram_config_reader_index);
    // DEBUG: act-mcast entry config -> ring buffer. 0xA0_SSFF  SS=sender_id, F=snd, F=rcv.
    WATCHER_RING_BUFFER_PUSH(
        0xA0000000u | ((act_mcast_sender_id & 0xff) << 8) | ((is_sender_core ? 1u : 0u) << 4) |
        (is_receiver_core ? 1u : 0u));

    // The act-mcast sender NoC-coord lookup table for the mcast dimension is supplied as positional
    // runtime varargs; get_vararg(act_w_outer_i) is the physical coord of sender act_w_outer_i.

    // Reader-indices base. On the resident (L1) path the config slice already lives in L1, reached by
    // base address from a local TensorAccessor (tensor::reader_indices) — no borrowed CB. On the
    // DRAM-config path it is DMA'd into a fresh L1 DFB (dfb::reader_indices) first.
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr;
#ifdef CONFIG_TENSOR_IN_DRAM
    DataflowBuffer cb_reader_indices_obj(dfb::reader_indices);
    packed_reader_indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_reader_indices_obj.get_write_ptr());
    {
        const auto config_accessor = TensorAccessor(tensor::reader_indices);
        constexpr uint32_t config_page_size = get_arg(args::config_page_size);
        noc.async_read(
            config_accessor, cb_reader_indices_obj, config_page_size, {.page_id = dram_config_reader_index}, {});
        noc.async_read_barrier();
        cb_reader_indices_obj.push_back(1);
    }
#else
    packed_reader_indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        (uint32_t)NOC_LOCAL_ADDR_OFFSET(TensorAccessor(tensor::reader_indices).get_noc_addr(0)));
    (void)dram_config_reader_index;
#endif

    // Set up receiver semaphore VALID value, to be mcasted to destinations after the data has been mcasted
    act_mcast_receiver_sem.set(VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast

    // TODO: need to make the read coalescing optimization cleaner
    // currently works for the case of num_coalesced_reads == weight_size_w since these reads are contiguous on both
    // src/dst side
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? weight_size_w * conv_act_c_read_bytes : conv_act_c_read_bytes);

    // Fully create act matrix and tilize it before mcast. The resident activation shard is reached by
    // L1 base address from a local TensorAccessor (tensor::act_sharded), not a borrowed self-loop CB.
    uint32_t act_l1_read_addr = (uint32_t)NOC_LOCAL_ADDR_OFFSET(TensorAccessor(tensor::act_sharded).get_noc_addr(0));

    experimental::set_read_state<coalesced_read_bytes>(noc, act_l1_read_addr);

    constexpr uint32_t window_outer_offset = padded_conv_act_size_w * conv_act_c_read_bytes * dilation_h;
    constexpr uint32_t stride_h_bytes = padded_conv_act_size_w * conv_act_c_read_bytes * dilation_h;
    constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;
    constexpr bool sliced_inner_dim = window_outer > 1;

    // Reset reader_idx to finish act_block_h_datums
    uint32_t reader_idx = 0;
    uint32_t start_reader_idx = 0;
    for (uint32_t nbh = 0; nbh < act_num_blocks_h; nbh++) {
        uint32_t reader_offset = act_l1_read_addr;
        for (uint32_t outer = 0; outer < window_outer; outer++) {
            reader_idx = start_reader_idx;
            cb_act_rm_obj.reserve_back(act_block_num_tiles_read);
            if (is_sender_core) {
                uint32_t l1_write_addr_act = cb_act_rm_obj.get_write_ptr();
                read_activation_data<
                    sliced_inner_dim,
                    dilation_w,
                    coalesced_read_bytes,
                    conv_act_c_read_bytes,
                    act_block_w_extra_align_bytes,
                    stride_w_bytes,
                    weight_size_w,
                    stride_w,
                    weight_size_h,
                    window_outer_offset>(
                    noc,
                    packed_reader_indices_ptr,
                    reader_offset,
                    l1_write_addr_act,
                    reader_idx,
                    act_l1_read_addr,
                    stride_h_bytes);
            }
            cb_act_rm_obj.push_back(act_block_num_tiles_read);

#ifndef SKIP_MCAST
            // Round robin self-mcast and receive tilized act matrix in cb_id_act
            // Compute should function like regular mm
            for (uint32_t act_w_outer_i = 0; act_w_outer_i < act_w_num_outer; act_w_outer_i++) {
                cb_act_obj.reserve_back(act_block_num_tiles);
                if (act_w_outer_i == act_mcast_sender_id) {
                    // MCAST SENDER: send entire tilized input to other cores in column
                    // wait until all act mcast destinations have atomically incremented the act semaphore_addr
                    // (i.e. its value should be act_mcast_num_dests), then reset the semaphore_addr value back to
                    // zero for the next block
                    WATCHER_RING_BUFFER_PUSH(0xA1000000u | RB_ITER(nbh, act_w_outer_i));  // sender: pre sem.wait
                    DPRINT("RDR Ssem nbh={} awo={}\n", (uint32_t)nbh, (uint32_t)act_w_outer_i);  // [#47797]
                    act_mcast_sender_sem.wait(act_mcast_num_dests + (is_receiver_core ? 0 : 1));
                    act_mcast_sender_sem.set(0);

                    act_mcast_receiver_sem.set(INVALID);

                    WATCHER_RING_BUFFER_PUSH(
                        0xA2000000u | RB_ITER(nbh, act_w_outer_i));  // sender: got bumps, pre mcast (waits tilized)
                    // [#47797] Sender got all bumps; about to mcast cb_tilized_in0. If this is the LAST
                    // reader line on a sender core, the compute never produced this nbh's tilized act.
                    DPRINT("RDR Smc nbh={} awo={}\n", (uint32_t)nbh, (uint32_t)act_w_outer_i);
                    mcast_block_chunked<
                        act_mcast_num_cores,
                        NOC_MAX_BURST_SIZE,
                        act_block_num_tiles,
                        act_mcast_tile_size_bytes>(
                        noc, mcast_ep, cb_tilized_in0_obj, is_receiver_core, cb_act_obj, act_mcast_rect);

                    WATCHER_RING_BUFFER_PUSH(0xA3000000u | RB_ITER(nbh, act_w_outer_i));  // sender: mcast data done
                    DPRINT("RDR Sdone nbh={} awo={}\n", (uint32_t)nbh, (uint32_t)act_w_outer_i);  // [#47797]

                    // Note: no need for write barrier, since these two multicasts are done on the same noc id and
                    // same vc even though cmd bufs are different Also, this only works because we are setting VCs
                    // statically (using NOC_CMD_STATIC_VC).

                    if (is_receiver_core) {
                        // We should also multicast VALID flag to destinations for receiver semaphore
                        if constexpr (act_mcast_num_cores) {
                            act_mcast_receiver_sem.set(VALID);
                            act_mcast_receiver_sem.set_multicast<NocOptions::MCAST_INCL_SRC>(
                                noc,
                                act_mcast_rect.noc_x_start,
                                act_mcast_rect.noc_y_start,
                                act_mcast_rect.noc_x_end,
                                act_mcast_rect.noc_y_end,
                                act_mcast_num_cores + 1);
                            noc.async_write_barrier();
                        }
                    } else {
                        act_mcast_receiver_sem.set(VALID);
                        act_mcast_receiver_sem.set_multicast(
                            noc,
                            act_mcast_rect.noc_x_start,
                            act_mcast_rect.noc_y_start,
                            act_mcast_rect.noc_x_end,
                            act_mcast_rect.noc_y_end,
                            act_mcast_num_cores + 1);
                    }
                } else if (is_receiver_core) {
                    // MCAST RECEIVER: receive entire tilized input from sender core
                    // Set act semaphore value to INVALID
                    act_mcast_receiver_sem.set(INVALID);

                    // Atomic increment source core counter.  The sender NoC coord for sender act_w_outer_i comes
                    // from the per-node runtime varargs lookup table.
                    if constexpr (transpose_mcast) {
                        act_mcast_sender_sem.up(noc, act_mcast_sender_noc_x, get_vararg(act_w_outer_i), 1);
                    } else {
                        act_mcast_sender_sem.up(noc, get_vararg(act_w_outer_i), act_mcast_sender_noc_x, 1);
                    }

                    WATCHER_RING_BUFFER_PUSH(
                        0xA5000000u | RB_ITER(nbh, act_w_outer_i));  // recv: bumped, pre wait VALID
                    DPRINT("RDR Rval nbh={} awo={}\n", (uint32_t)nbh, (uint32_t)act_w_outer_i);  // [#47797]
                    // wait on act semaphore value to become VALID (set by mcast sender after it multicasts data)
                    act_mcast_receiver_sem.wait(VALID);
                    WATCHER_RING_BUFFER_PUSH(0xA6000000u | RB_ITER(nbh, act_w_outer_i));  // recv: got VALID
                    DPRINT("RDR Rgot nbh={} awo={}\n", (uint32_t)nbh, (uint32_t)act_w_outer_i);  // [#47797]
                }
                cb_act_obj.push_back(act_block_num_tiles);
            }  // act_w_num_outer

            cb_tilized_in0_obj.pop_front(act_block_num_tiles);
#endif
        }
        start_reader_idx = reader_idx;
        if constexpr (split_reader_enabled) {
            // Increment reader index for the next number of segments (number of segments for other reader)
            // Only read reader indices on cores that have sharded input (is_sender_core).
            if (is_sender_core) {
                start_reader_idx += (static_cast<uint32_t>(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1);
            }
        }
    }

    noc.async_write_barrier();
}
