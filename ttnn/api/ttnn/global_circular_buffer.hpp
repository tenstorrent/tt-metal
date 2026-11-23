// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <vector>
#include <tt-metalium/global_circular_buffer.hpp>
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/prefetcher_pipe.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::global_circular_buffer {

// Multi Device APIs
GlobalCircularBuffer create_global_circular_buffer(
    MeshDevice* mesh_device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

// The weight <-> matmul contract of a gather-in0 or mcast-in0 1D matmul reading a DRAM weight's
// K-blocks from `gcb`, returning the block count: gather returns the receiver/ring count; mcast returns
// `weight_K_tiles / program_config.in0_block_w`. The Tensor prefetcher delivers into PrefetcherPipes
// (the overload below), not into a GlobalCircularBuffer; the matmul's global_cb validation still runs
// these checks. The weight's DRAM layout is detected and the matching cross-checks run:
//   * receiver-contiguous: an NdShardSpec DRAM tensor with one full-K, N/receiver_count shard per
//     receiver;
//   * legacy K-row-major: a WIDTH_SHARDED DRAM tensor with one full-K, N/num_senders shard per bank,
//     each bank's per-receiver N split evenly across its num_global_cb_receivers cores, and a `gcb`
//     whose senders each drive exactly num_global_cb_receivers receivers (this layout's sender slices
//     its shard by one receivers-per-bank number, so an uneven fan-out delivers the wrong pages);
// and, for either layout:
//   * weight K-tiles divides the consumer's block count exactly;
//   * the weight's per-receiver N (in tiles) equals program_config.per_core_N — otherwise the
//     prefetcher's pushed page size and the matmul's in1 remote-CB page size disagree and the page-credit
//     accounting desyncs.
// Mcast uses natural FIFO order and therefore requires `stream_in1=false`.
uint32_t tensor_prefetcher_block_count_for_matmul_1d(
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
    const ttnn::Tensor& weight,
    const GlobalCircularBuffer& gcb);

// Compute the validated `block_count` to pair with `weight` in a TensorPrefetcherInput when feeding a
// gather-in0 or mcast-in0 1D matmul from DRAM-sender PrefetcherPipes. It enforces the same weight <->
// matmul contract as the overload above; only the object that names the receivers differs, and for
// gather-in0 a receiver-contiguous shard may run past the weight's K, so that a ring whose last in0
// shards are padded can cut it into whole shard-deep K-blocks. A legacy K-row-major weight takes its
// receivers per bank from the pipes rather than from num_global_cb_receivers, and needs one pipe per
// bank (support_multi_receiver_shards), each with the same receiver count.
uint32_t tensor_prefetcher_block_count_for_matmul_1d(
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
    const ttnn::Tensor& weight,
    const ttnn::PrefetcherPipeList& prefetcher_pipes);

}  // namespace ttnn::global_circular_buffer
