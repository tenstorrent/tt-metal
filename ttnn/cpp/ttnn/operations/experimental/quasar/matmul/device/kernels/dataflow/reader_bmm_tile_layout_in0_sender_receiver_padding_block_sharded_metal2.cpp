// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp.
//
// Algorithm body matches the legacy kernel; only the host-binding surface is converted to Metal 2.0:
//   - positional get_compile_time_arg_val(N)  -> named get_arg(args::name)
//   - the leading in0_mcast_noc_x / in0_mcast_noc_y arrays (legacy get_arg_addr scratch) -> runtime
//     varargs accessed positionally via get_vararg()
//   - named CB-index CTAs ("cb_in0","cb_in0_sharded") -> dfb:: tokens
//   - Semaphore<>(get_compile_time_arg_val(id)) -> Semaphore(sem::name)
//
// This is the in0-sharded sender/receiver variant. `core_has_output_block_work` /
// `core_in_in0_receiver_mcast_grid` are the two leading named CTAs the factory patches per
// kernel-variant (sender / no-work-in-receiver / no-work-not-in-receiver).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "api/debug/dprint.h"  // [DEBUG #47797] in0 mcast handshake diagnosis
#include "api/debug/ring_buffer.h"  // [DEBUG #47797] PREBARRIER counters (DPRINT MMIO path unsupported on craq-sim)

void kernel_main() {
    constexpr bool core_has_output_block_work = (bool)get_arg(args::core_has_output_block_work);
    constexpr bool core_in_in0_receiver_mcast_grid = (bool)get_arg(args::core_in_in0_receiver_mcast_grid);

    constexpr uint32_t in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    constexpr uint32_t in0_block_size_bytes = get_arg(args::in0_block_size_bytes);
    constexpr uint32_t in0_last_ktile_w = get_arg(args::in0_last_ktile_w);
    constexpr uint32_t in0_last_ktile_h = get_arg(args::in0_last_ktile_h);

    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_arg(args::num_blocks_inner_dim);
    constexpr uint32_t num_blocks_w_dim = get_arg(args::num_blocks_w_dim);
    constexpr uint32_t num_blocks_h_dim = get_arg(args::num_blocks_h_dim);
    // in0 mcast args
    constexpr uint32_t in0_mcast_num_dests = get_arg(args::in0_mcast_num_dests);
    constexpr uint32_t in0_mcast_num_cores = get_arg(args::in0_mcast_num_cores);
    constexpr uint32_t num_x = get_arg(args::num_x);
    constexpr uint32_t num_y = get_arg(args::num_y);
    constexpr bool transpose_mcast = (bool)get_arg(args::transpose_mcast);
    constexpr uint32_t shard_width_in_tiles = get_arg(args::shard_width_in_tiles);
    constexpr uint32_t shard_height_in_tiles = get_arg(args::shard_height_in_tiles);
    constexpr uint32_t in0_block_w = get_arg(args::in0_block_w);
    constexpr uint32_t in0_block_h = get_arg(args::in0_block_h);

    constexpr uint32_t batch = get_arg(args::batch);
    constexpr bool fuse_op = (bool)get_arg(args::fuse_op);

    const uint32_t sender_id = get_arg(args::sender_id);
    const uint32_t in0_mcast_dest_noc_start_x = get_arg(args::in0_mcast_dest_noc_start_x);
    const uint32_t in0_mcast_dest_noc_start_y = get_arg(args::in0_mcast_dest_noc_start_y);
    const uint32_t in0_mcast_dest_noc_end_x = get_arg(args::in0_mcast_dest_noc_end_x);
    const uint32_t in0_mcast_dest_noc_end_y = get_arg(args::in0_mcast_dest_noc_end_y);
    // in0_mcast_noc_x[num_x] then in0_mcast_noc_y[num_y] are runtime varargs.
    uint32_t vararg_idx = 0;
    uint32_t in0_mcast_noc_x[num_x];
    uint32_t in0_mcast_noc_y[num_y];
    for (uint32_t i = 0; i < num_x; ++i) {
        in0_mcast_noc_x[i] = get_vararg(vararg_idx++);
    }
    for (uint32_t i = 0; i < num_y; ++i) {
        in0_mcast_noc_y[i] = get_vararg(vararg_idx++);
    }
    // Any fused-op signaler varargs follow; rt_args_idx tracks the named-RTA tail for MatmulOpReceiver.
    uint32_t rt_args_idx = vararg_idx;

    constexpr uint32_t cb_id_in0 = dfb::cb_in0;

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr DataFormat in0_data_format = get_dataformat(cb_id_in0);

    constexpr uint32_t num_blocks_per_shard = shard_width_in_tiles / in0_block_w;
    constexpr bool extract_shard_sub_blocks = shard_height_in_tiles > 1 && num_blocks_per_shard > 1;
    constexpr uint32_t out_block_h = shard_height_in_tiles / num_blocks_h_dim;
    constexpr uint32_t shard_read_stride = shard_width_in_tiles * in0_single_tile_size_bytes;
    constexpr uint32_t shard_read_width = in0_single_tile_size_bytes * in0_block_w;
    constexpr uint32_t in0_tensor_next_h_dim_block_stride = shard_read_stride * in0_block_h;

    Noc noc;
    DataflowBuffer cb_in0(cb_id_in0);
    Semaphore sender_sem(sem::in0_sender);
    Semaphore receiver_sem(sem::in0_receiver);

    constexpr uint32_t num_remote_senders = (num_blocks_inner_dim + num_blocks_per_shard - 1) / num_blocks_per_shard;
    uint32_t remote_sender_noc_x[num_remote_senders];
    uint32_t remote_sender_noc_y[num_remote_senders];
    if constexpr (transpose_mcast) {
        uint32_t x = 0, y = 0;
        for (uint32_t i = 0; i < num_remote_senders; ++i) {
            remote_sender_noc_x[i] = in0_mcast_noc_x[x];
            remote_sender_noc_y[i] = in0_mcast_noc_y[y];
            ++y;
            if (y == num_y) {
                y = 0;
                ++x;
            }
        }
    } else {
        uint32_t x = 0, y = 0;
        for (uint32_t i = 0; i < num_remote_senders; ++i) {
            remote_sender_noc_x[i] = in0_mcast_noc_x[x];
            remote_sender_noc_y[i] = in0_mcast_noc_y[y];
            ++x;
            if (x == num_x) {
                x = 0;
                ++y;
            }
        }
    }
    receiver_sem.set(VALID);

    // [DEBUG mcast2d hang] Watcher-reliable (unlike DPRINT on craq-sim) dump of the mcast axis args the
    // kernel actually received: marker 0x5E4D0000, num_x, num_y, in0_mcast_num_dests, and the first two
    // resolved remote-sender coords. If num_x==1/num_y>1 (or remote_sender0/1 share an x and differ in y)
    // the receiver is acking a COLUMN while the sender counts num_dests along the row -> the deadlock.
    WATCHER_RING_BUFFER_PUSH(0x5E4D0000u);
    WATCHER_RING_BUFFER_PUSH((uint32_t)num_x);
    WATCHER_RING_BUFFER_PUSH((uint32_t)num_y);
    WATCHER_RING_BUFFER_PUSH((uint32_t)in0_mcast_num_dests);
    WATCHER_RING_BUFFER_PUSH(remote_sender_noc_x[0]);
    WATCHER_RING_BUFFER_PUSH(remote_sender_noc_y[0]);
    if constexpr (num_remote_senders > 1) {
        WATCHER_RING_BUFFER_PUSH(remote_sender_noc_x[1]);
        WATCHER_RING_BUFFER_PUSH(remote_sender_noc_y[1]);
    }

    // ---- [DEBUG #47797] one-shot per-core dump of the in0 mcast handshake config. ----
    // For block 0, block_id == 0, so the core whose sender_id == 0 must take the sender branch
    // (line `if (block_id == sender_id)`). If NO core prints sender_id==0, nobody multicasts VALID
    // and every receiver hangs at `receiver_sem.wait(VALID)`. Compare `sid` here against the host
    // log_info "[in0bs-host]" line for the same core; they must agree (sid == core.x non-transpose).
    DPRINT(
        "[in0bs-dev] sid={} noc={} cir={} tm={} nbps={} nblk={} ndst={} ncore={} nx={} ny={}\n",
        (uint32_t)sender_id,
        (uint32_t)noc_index,  // [DEBUG #47797] actual NOC the framework launched this kernel on
        (uint32_t)core_in_in0_receiver_mcast_grid,
        (uint32_t)transpose_mcast,
        (uint32_t)num_blocks_per_shard,
        (uint32_t)num_blocks_inner_dim,
        (uint32_t)in0_mcast_num_dests,
        (uint32_t)in0_mcast_num_cores,
        (uint32_t)num_x,
        (uint32_t)num_y);
    DPRINT(
        "[in0bs-dev] dst_x[{}..{}] dst_y[{}..{}] remote_sender0=({},{})\n",
        (uint32_t)in0_mcast_dest_noc_start_x,
        (uint32_t)in0_mcast_dest_noc_end_x,
        (uint32_t)in0_mcast_dest_noc_start_y,
        (uint32_t)in0_mcast_dest_noc_end_y,
        (uint32_t)remote_sender_noc_x[0],
        (uint32_t)remote_sender_noc_y[0]);

    // The resident in0 shard is reached by L1 base address from a local TensorAccessor over the in0
    // tensor (no borrowed self-loop CB, which Metal 2.0 forbids on DM kernels).
    uint32_t in0_tensor_shard_read_addr = (uint32_t)NOC_LOCAL_ADDR_OFFSET(TensorAccessor(tensor::in0).get_noc_addr(0));
    uint32_t in0_tensor_read_addr = 0;

    MatmulOpReceiver fused_op_receiver;
    if constexpr (fuse_op) {
        fused_op_receiver = MatmulOpReceiver(
            sender_id < num_remote_senders, /* wait_for_op_signal */
            rt_args_idx,
            num_blocks_inner_dim,
            in0_block_w /* tiles_per_block (in the same dimension as tensor slice) */
        );
    }

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t in0_tensor_current_h_dim_block_start_addr = in0_tensor_shard_read_addr;
        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                uint32_t in0_tensor_current_inner_dim_block_start_addr = in0_tensor_current_h_dim_block_start_addr;
                for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                    uint32_t block_id = block / num_blocks_per_shard;
                    if constexpr (fuse_op) {
                        block_id = fused_op_receiver.align_to_slice_and_sync(block, sender_id);
                    }

                    // [DEBUG #47797] Per-block enter trace. role=1 => this core is the sender for
                    // this block (multicasts in0+VALID); role=0 => receiver (waits for VALID). The
                    // LAST "enter" line per core file shows the block it is stuck on; the matching
                    // "SENT" line (below) for that block tells us whether its sender actually issued
                    // the multicast. Print only first batch/h/w slice to bound spam to <=num_blocks lines.
                    if (b == 0 && bh == 0 && bw == 0) {
                        DPRINT(
                            "[in0bs-dev] enter block={} block_id={} sid={} role={}\n",
                            (uint32_t)block,
                            (uint32_t)block_id,
                            (uint32_t)sender_id,
                            (uint32_t)(block_id == sender_id));
                    }

                    cb_in0.reserve_back(in0_block_num_tiles);

                    if constexpr (core_in_in0_receiver_mcast_grid) {
                        // Set in0 semaphore value to INVALID
                        receiver_sem.set(INVALID);
                    }

                    if (block_id == sender_id) {
                        // Operand 0
                        uint32_t in0_tensor_local_l1_write_addr = cb_in0.get_write_ptr();

                        if constexpr (extract_shard_sub_blocks) {
                            in0_tensor_read_addr = in0_tensor_local_l1_write_addr;

                            uint32_t l1_write_extract_shard_in0 = in0_tensor_local_l1_write_addr;
                            UnicastEndpoint self_ep;
                            uint32_t noc_shard_read_l1_addr = in0_tensor_current_inner_dim_block_start_addr;

                            for (uint32_t i = 0; i < out_block_h; i++) {
                                noc.async_read(
                                    self_ep,
                                    CoreLocalMem<uint32_t>(l1_write_extract_shard_in0),
                                    shard_read_width,
                                    {.noc_x = my_x[0], .noc_y = my_y[0], .addr = noc_shard_read_l1_addr},
                                    {});
                                l1_write_extract_shard_in0 += shard_read_width;
                                noc_shard_read_l1_addr += shard_read_stride;
                            }

                            in0_tensor_current_inner_dim_block_start_addr += shard_read_width;

                            noc.async_read_barrier();

                            if constexpr (in0_last_ktile_w > 0) {
                                if ((block == num_blocks_inner_dim - 1)) {
                                    for (uint32_t h = 0; h < out_block_h; ++h) {
                                        auto in0_last_ktile_w_ptr =
                                            in0_tensor_read_addr +
                                            (h * in0_block_w + in0_block_w - 1) * in0_single_tile_size_bytes;
                                        pad_last_ktile<in0_data_format, in0_last_ktile_w>(in0_last_ktile_w_ptr);
                                    }
                                }
                            }
                            if constexpr (in0_last_ktile_h > 0) {
                                if ((block == num_blocks_inner_dim - 1)) {
                                    for (uint32_t w = 0; w < in0_block_w; ++w) {
                                        auto in0_last_ktile_h_ptr =
                                            in0_tensor_read_addr +
                                            (out_block_h - 1) * in0_block_w * in0_single_tile_size_bytes +
                                            w * in0_single_tile_size_bytes;
                                        pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(
                                            in0_last_ktile_h_ptr);
                                    }
                                }
                            }
                        } else {
                            in0_tensor_read_addr = in0_tensor_current_inner_dim_block_start_addr;
                            in0_tensor_current_inner_dim_block_start_addr += in0_block_size_bytes;

                            if constexpr (in0_last_ktile_w > 0) {
                                if ((block == num_blocks_inner_dim - 1)) {
                                    for (uint32_t h = 0; h < in0_block_h; ++h) {
                                        auto in0_last_ktile_w_ptr =
                                            in0_tensor_read_addr +
                                            (h * in0_block_w + in0_block_w - 1) * in0_single_tile_size_bytes;
                                        pad_last_ktile<in0_data_format, in0_last_ktile_w>(in0_last_ktile_w_ptr);
                                    }
                                }
                            }
                            if constexpr (in0_last_ktile_h > 0) {
                                if ((block == num_blocks_inner_dim - 1)) {
                                    for (uint32_t w = 0; w < in0_block_w; ++w) {
                                        auto in0_last_ktile_h_ptr =
                                            in0_tensor_read_addr +
                                            ((in0_block_h - 1) * in0_block_w + w) * in0_single_tile_size_bytes;
                                        pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(
                                            in0_last_ktile_h_ptr);
                                    }
                                }
                            }
                        }

                        // [DEBUG #47797] SENDER pre-wait: marker 0x5E4D0001, noc, sender_sem value,
                        // expected (num_dests[-1]). If the newest ring-buffer entry is this and the value
                        // never reaches expected, receivers' sender_sem.up acks aren't arriving at the sender.
                        // [DISABLED — handshake confirmed working (sems reach 7 & 3); these loop markers
                        // flood the 32-entry ring buffer and hide the compute-side stall. Re-enable if the
                        // handshake regresses.]
#if 0
                        WATCHER_RING_BUFFER_PUSH(0x5E4D0001u);
                        WATCHER_RING_BUFFER_PUSH((uint32_t)noc_index);
                        WATCHER_RING_BUFFER_PUSH((uint32_t)sender_sem.get_value());
                        WATCHER_RING_BUFFER_PUSH((uint32_t)(core_in_in0_receiver_mcast_grid ? in0_mcast_num_dests - 1
                                                                                            : in0_mcast_num_dests));
#endif
                        if constexpr (core_in_in0_receiver_mcast_grid) {
                            // wait for every core in receiver grid EXCLUDING myself
                            sender_sem.wait(in0_mcast_num_dests - 1);
                        } else {
                            // wait for every core in receiver grid
                            sender_sem.wait(in0_mcast_num_dests);
                        }
                        sender_sem.set(0);

                        // Now we have the block in the CB address, we can mcast to dests!
                        if constexpr (core_in_in0_receiver_mcast_grid) {
                            // Mcast from/to same CB
                            if constexpr (extract_shard_sub_blocks) {
                                if constexpr (in0_mcast_num_cores > 1) {
                                    MulticastEndpoint mcast_dst;
                                    noc.async_write_multicast(
                                        CoreLocalMem<uint32_t>(in0_tensor_read_addr),
                                        mcast_dst,
                                        in0_block_size_bytes,
                                        in0_mcast_num_cores - 1,
                                        {},
                                        {.noc_x_start = in0_mcast_dest_noc_start_x,
                                         .noc_y_start = in0_mcast_dest_noc_start_y,
                                         .noc_x_end = in0_mcast_dest_noc_end_x,
                                         .noc_y_end = in0_mcast_dest_noc_end_y,
                                         .addr = in0_tensor_local_l1_write_addr},
                                        true);
                                }
                            }
                            // Mcast from different CB to another CB
                            else {
                                if constexpr (in0_mcast_num_cores == 1) {
                                    UnicastEndpoint ucast_dst;
                                    noc.async_write(
                                        CoreLocalMem<uint32_t>(in0_tensor_read_addr),
                                        ucast_dst,
                                        in0_block_size_bytes,
                                        {},
                                        {.noc_x = in0_mcast_dest_noc_start_x,
                                         .noc_y = in0_mcast_dest_noc_start_y,
                                         .addr = in0_tensor_local_l1_write_addr});
                                } else {
                                    MulticastEndpoint mcast_dst;
                                    noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                                        CoreLocalMem<uint32_t>(in0_tensor_read_addr),
                                        mcast_dst,
                                        in0_block_size_bytes,
                                        in0_mcast_num_cores,
                                        {},
                                        {.noc_x_start = in0_mcast_dest_noc_start_x,
                                         .noc_y_start = in0_mcast_dest_noc_start_y,
                                         .noc_x_end = in0_mcast_dest_noc_end_x,
                                         .noc_y_end = in0_mcast_dest_noc_end_y,
                                         .addr = in0_tensor_local_l1_write_addr},
                                        true);
                                }
                            }

                            // We should also multicast the flag to destinations
                            receiver_sem.set(VALID);
                            if constexpr (in0_mcast_num_cores > 1) {
                                receiver_sem.set_multicast<NocOptions::MCAST_INCL_SRC>(
                                    noc,
                                    in0_mcast_dest_noc_start_x,
                                    in0_mcast_dest_noc_start_y,
                                    in0_mcast_dest_noc_end_x,
                                    in0_mcast_dest_noc_end_y,
                                    in0_mcast_num_cores);
                            }
                        } else {
                            // If we are not part of receiver grid, always do a regular mcast to all cores
                            MulticastEndpoint mcast_dst;
                            noc.async_write_multicast(
                                CoreLocalMem<uint32_t>(in0_tensor_read_addr),
                                mcast_dst,
                                in0_block_size_bytes,
                                in0_mcast_num_cores,
                                {},
                                {.noc_x_start = in0_mcast_dest_noc_start_x,
                                 .noc_y_start = in0_mcast_dest_noc_start_y,
                                 .noc_x_end = in0_mcast_dest_noc_end_x,
                                 .noc_y_end = in0_mcast_dest_noc_end_y,
                                 .addr = in0_tensor_local_l1_write_addr},
                                true);

                            // We should also multicast the flag to destinations
                            receiver_sem.set(VALID);
                            receiver_sem.set_multicast(
                                noc,
                                in0_mcast_dest_noc_start_x,
                                in0_mcast_dest_noc_start_y,
                                in0_mcast_dest_noc_end_x,
                                in0_mcast_dest_noc_end_y,
                                in0_mcast_num_cores);
                        }

                        if constexpr (!(core_in_in0_receiver_mcast_grid && (in0_mcast_num_cores == 1))) {
                            noc.async_writes_flushed();
                        }
                        // [DEBUG #47797] Sender finished issuing the in0 data + VALID multicast for
                        // this block on `noc`. If this prints for the stuck block but the receivers
                        // never advance past it, the multicast/flag is not landing (NOC/geometry),
                        // not an arg problem. (b==0,bh==0,bw==0 slice only, to bound spam.)
                        if (b == 0 && bh == 0 && bw == 0) {
                            DPRINT(
                                "[in0bs-dev] SENT block={} noc={} ncore={} dst_x[{}..{}] dst_y[{}..{}]\n",
                                (uint32_t)block,
                                (uint32_t)noc_index,
                                (uint32_t)in0_mcast_num_cores,
                                (uint32_t)in0_mcast_dest_noc_start_x,
                                (uint32_t)in0_mcast_dest_noc_end_x,
                                (uint32_t)in0_mcast_dest_noc_start_y,
                                (uint32_t)in0_mcast_dest_noc_end_y);
                        }
                    } else if constexpr (core_in_in0_receiver_mcast_grid) {
                        // [DEBUG #47797] RECEIVER ack: marker 0x5E4D0002, noc, block_id, the sender coords
                        // this .up targets. If the sender is stuck at 0x5E4D0001, check these coords resolve
                        // to the actual sender core.
                        // [DISABLED — handshake confirmed working; see note at 0x5E4D0001 above.]
#if 0
                        WATCHER_RING_BUFFER_PUSH(0x5E4D0002u);
                        WATCHER_RING_BUFFER_PUSH((uint32_t)noc_index);
                        WATCHER_RING_BUFFER_PUSH((uint32_t)block_id);
                        WATCHER_RING_BUFFER_PUSH((uint32_t)remote_sender_noc_x[block_id]);
                        WATCHER_RING_BUFFER_PUSH((uint32_t)remote_sender_noc_y[block_id]);
#endif
                        // Increment remote sender's semaphore using pre-computed coordinates
                        sender_sem.up(noc, remote_sender_noc_x[block_id], remote_sender_noc_y[block_id], 1);
                    }

                    if constexpr (core_in_in0_receiver_mcast_grid) {
                        // [DEBUG #47797] RECEIVER pre-wait: marker 0x5E4D0003, noc, receiver_sem value, VALID.
                        // If the newest ring-buffer entry is this and the value never reaches VALID, the
                        // sender's receiver_sem.set_multicast(VALID) isn't reaching this receiver.
                        // [DISABLED — handshake confirmed working; see note at 0x5E4D0001 above.]
#if 0
                        WATCHER_RING_BUFFER_PUSH(0x5E4D0003u);
                        WATCHER_RING_BUFFER_PUSH((uint32_t)noc_index);
                        WATCHER_RING_BUFFER_PUSH((uint32_t)receiver_sem.get_value());
                        WATCHER_RING_BUFFER_PUSH((uint32_t)VALID);
#endif
                        // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
                        receiver_sem.wait(VALID);
                    }
                    cb_in0.push_back(in0_block_num_tiles);

                    if constexpr (!core_has_output_block_work) {
                        cb_in0.pop_front(in0_block_num_tiles);
                    }
                }
            }
            in0_tensor_current_h_dim_block_start_addr += in0_tensor_next_h_dim_block_stride;
        }
    }

    // [DEBUG #47797] Dump SW NoC issued-counters vs HW completion just before the drain. The
    // full_barrier hangs in the reads-flushed wait (NIU_MST_RD_RESP_RECEIVED == noc_reads_num_issued).
    // If reads_issued is non-zero here (this kernel issues no source-level reads when
    // extract_shard_sub_blocks is false), a metal2 primitive over-incremented the issued counter.
    // reads_flushed / npw_sent == 0 means that category will block the barrier forever.
    DPRINT(
        "[in0bs-dev] PREBARRIER noc={} reads_issued={} npw_issued={} reads_flushed={} npw_sent={}\n",
        (uint32_t)noc_index,
        (uint32_t)noc_reads_num_issued[noc_index],
        (uint32_t)noc_nonposted_writes_num_issued[noc_index],
        (uint32_t)ncrisc_noc_reads_flushed(noc_index),
        (uint32_t)ncrisc_noc_nonposted_writes_sent(noc_index));
    // [DEBUG #47797] PREBARRIER counter ring-buffer dump REMOVED: the barrier isn't the hang
    // (scmdbuf_tr_ack is stubbed to 0 in craq-sim, so async_full_barrier passes). The handshake dumps
    // are now at the sender_sem.wait / receiver_sem.wait / sender_sem.up sites (markers 0x5E4D000{1,2,3}).

    noc.async_write_barrier();
    // [#47797] Fully drain this kernel's NOC transactions before completing. The metal2 firmware
    // epilogue does not auto-flush, so a kernel that issues mcast writes + non-posted atomics
    // (sender_sem.up / receiver_sem mcast) and (in the extract path) reads must drain all NOC
    // categories itself, or the watcher trips DebugAssertNCriscNOCReadsFlushedTripped (inter-kernel
    // race). Matches the sibling metal2 readers (in0_sender_padding, in1_sender_writer).
    noc.async_full_barrier();
}
