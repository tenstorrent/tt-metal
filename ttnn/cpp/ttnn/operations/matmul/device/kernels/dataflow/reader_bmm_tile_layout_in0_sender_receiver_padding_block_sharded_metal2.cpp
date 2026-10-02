// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp, which
// lives beside it. Factories ported to Metal 2.0 bind this fork; the original serves the consumers
// still on the legacy ProgramDescriptor API. Until the last of them migrates and the original is
// retired, changes to either copy likely belong in the other too.
//
// The binding and argument names below are this fork's interface: every factory that later ports
// onto it inherits them and cannot rename them.
//
// The rotating in0 multicast is the helper's named block (MCAST_ARGS(in0)): the host attaches
// the sender schedule, receiver roles, and semaphores, so this kernel carries no coordinates.
//
// The CCL fused-op receiver is gated behind FUSE_OP rather than a compile-time arg: MatmulOpReceiver
// consumes *positional* runtime args through an index it advances by reference, and it lives outside
// this op's directory (ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp), so it cannot be fed
// from named arguments without changing a file this port may not touch.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#ifdef FUSE_OP
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#endif
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args_metal2.hpp"

void kernel_main() {
    constexpr bool core_has_output_block_work = static_cast<bool>(get_arg(args::core_has_output_block_work));

    constexpr auto in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    constexpr auto in0_block_size_bytes = get_arg(args::in0_block_size_bytes);
    constexpr auto in0_last_ktile_w = get_arg(args::in0_last_ktile_w);
    constexpr auto in0_last_ktile_h = get_arg(args::in0_last_ktile_h);

    // in0/in1 common args
    constexpr auto num_blocks_inner_dim = get_arg(args::num_blocks_inner_dim);
    constexpr auto num_blocks_w_dim = get_arg(args::num_blocks_w_dim);
    constexpr auto num_blocks_h_dim = get_arg(args::num_blocks_h_dim);
    constexpr auto shard_width_in_tiles = get_arg(args::shard_width_in_tiles);
    constexpr auto shard_height_in_tiles = get_arg(args::shard_height_in_tiles);
    constexpr auto in0_block_w = get_arg(args::in0_block_w);
    constexpr auto in0_block_h = get_arg(args::in0_block_h);

    constexpr auto batch = get_arg(args::batch);

#ifdef FUSE_OP
    uint32_t rt_args_idx = 0;
#endif
    [[maybe_unused]] const uint32_t sender_id = get_arg(args::sender_id);

    // in0 is filled here from this core's shard (or from a remote sender's multicast) and drained by
    // the compute kernel; in0_sharded is the resident shard itself, borrowed onto the tensor memory.
    constexpr auto dfb_id_in0 = dfb::in0;
    constexpr auto dfb_id_in2 = dfb::in0_sharded;

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(dfb_id_in0);
    constexpr DataFormat in0_data_format = get_dataformat(dfb_id_in0);

    constexpr uint32_t num_blocks_per_shard = shard_width_in_tiles / in0_block_w;
    // In case we need to send multiple blocks per shard, and shard height in tiles is greater than 1
    // Than we first need to extract the sub-blocks from the shard, and then send them to the destinations
    constexpr bool extract_shard_sub_blocks = shard_height_in_tiles > 1 && num_blocks_per_shard > 1;
    constexpr uint32_t out_block_h = shard_height_in_tiles / num_blocks_h_dim;
    constexpr uint32_t shard_read_stride = shard_width_in_tiles * in0_single_tile_size_bytes;
    constexpr uint32_t shard_read_width = in0_single_tile_size_bytes * in0_block_w;
    constexpr uint32_t in0_tensor_next_h_dim_block_stride = shard_read_stride * in0_block_h;

    constexpr uint32_t num_remote_senders = (num_blocks_inner_dim + num_blocks_per_shard - 1) / num_blocks_per_shard;

    constexpr auto in0_mcast_args = MCAST_ARGS(in0);
    static_assert(in0_mcast_args.active);
    static_assert(num_remote_senders <= in0_mcast_args.num_senders);

    const Noc noc;
    DataflowBuffer dfb_in0(dfb_id_in0);
    DataflowBuffer dfb_in2(dfb_id_in2);
    auto in0_sender_pipe = in0_mcast_args.optional_sender(noc);
    auto in0_receiver_pipe = in0_mcast_args.optional_receiver(noc);
    const bool can_receive = in0_mcast_args.can_receive();

    dfb_in2.reserve_back(batch * in0_block_num_tiles);

    const uint32_t in0_tensor_shard_read_addr = dfb_in2.get_read_ptr();
    uint32_t in0_tensor_read_addr = 0;

#ifdef FUSE_OP
    MatmulOpReceiver fused_op_receiver = MatmulOpReceiver(
        sender_id < num_remote_senders, /* wait_for_op_signal */
        rt_args_idx,
        num_blocks_inner_dim,
        in0_block_w /* tiles_per_block (in the same dimension as tensor slice) */
    );
#endif

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t in0_tensor_current_h_dim_block_start_addr = in0_tensor_shard_read_addr;
        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                uint32_t in0_tensor_current_inner_dim_block_start_addr = in0_tensor_current_h_dim_block_start_addr;
                for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                    uint32_t block_id = block / num_blocks_per_shard;
                    // If used fused op, make block_id conform to ordering of tensor slices from all
                    // gather
#ifdef FUSE_OP
                    block_id = fused_op_receiver.align_to_slice_and_sync(block, sender_id);
#endif

                    dfb_in0.reserve_back(in0_block_num_tiles);

                    if (in0_mcast_args.should_send(block_id)) {
                        // Operand 0
                        const uint32_t in0_tensor_local_l1_write_addr = dfb_in0.get_write_ptr();

                        if constexpr (extract_shard_sub_blocks) {
                            in0_tensor_read_addr = in0_tensor_local_l1_write_addr;

                            uint32_t l1_write_extract_shard_in0 = in0_tensor_local_l1_write_addr;
                            const UnicastEndpoint self_ep;
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
                                if (block == num_blocks_inner_dim - 1) {
                                    for (uint32_t h = 0; h < out_block_h; ++h) {
                                        auto in0_last_ktile_w_ptr =
                                            in0_tensor_read_addr +
                                            ((h * in0_block_w + in0_block_w - 1) * in0_single_tile_size_bytes);
                                        pad_last_ktile<in0_data_format, in0_last_ktile_w>(in0_last_ktile_w_ptr);
                                    }
                                }
                            }
                            if constexpr (in0_last_ktile_h > 0) {
                                if (block == num_blocks_inner_dim - 1) {
                                    for (uint32_t w = 0; w < in0_block_w; ++w) {
                                        auto in0_last_ktile_h_ptr =
                                            in0_tensor_read_addr +
                                            ((out_block_h - 1) * in0_block_w * in0_single_tile_size_bytes) +
                                            (w * in0_single_tile_size_bytes);
                                        pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(
                                            in0_last_ktile_h_ptr);
                                    }
                                }
                            }
                        } else {
                            in0_tensor_read_addr = in0_tensor_current_inner_dim_block_start_addr;
                            in0_tensor_current_inner_dim_block_start_addr += in0_block_size_bytes;

                            if constexpr (in0_last_ktile_w > 0) {
                                if (block == num_blocks_inner_dim - 1) {
                                    for (uint32_t h = 0; h < in0_block_h; ++h) {
                                        auto in0_last_ktile_w_ptr =
                                            in0_tensor_read_addr +
                                            ((h * in0_block_w + in0_block_w - 1) * in0_single_tile_size_bytes);
                                        pad_last_ktile<in0_data_format, in0_last_ktile_w>(in0_last_ktile_w_ptr);
                                    }
                                }
                            }
                            if constexpr (in0_last_ktile_h > 0) {
                                if (block == num_blocks_inner_dim - 1) {
                                    for (uint32_t w = 0; w < in0_block_w; ++w) {
                                        auto in0_last_ktile_h_ptr =
                                            in0_tensor_read_addr +
                                            (((in0_block_h - 1) * in0_block_w + w) * in0_single_tile_size_bytes);
                                        pad_last_transposed_ktile<in0_data_format, in0_last_ktile_h>(
                                            in0_last_ktile_h_ptr);
                                    }
                                }
                            }
                        }

                        in0_sender_pipe->send(
                            in0_tensor_read_addr, in0_tensor_local_l1_write_addr, in0_block_size_bytes);
                    } else if (can_receive) {
                        in0_receiver_pipe->receive(block_id);
                    }
                    dfb_in0.push_back(in0_block_num_tiles);

                    // If core does not produce output block work, free the in0 buffer immediately.
                    // This is necessary since mcast is in lockstep; this ensures write ptr addresses are synced
                    // properly for cores that only send and have no compute / writer active. Technically, don't have to
                    // do this if the in0 buffer is not double buffered.
                    if constexpr (!core_has_output_block_work) {
                        dfb_in0.pop_front(in0_block_num_tiles);
                    }
                }
            }
            in0_tensor_current_h_dim_block_start_addr += in0_tensor_next_h_dim_block_stride;
        }
    }

    noc.async_write_barrier();
}
