// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <tt-metalium/global_circular_buffer.hpp>
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::global_circular_buffer {

// Single Device APIs
GlobalCircularBuffer create_global_circular_buffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

// Multi Device APIs
GlobalCircularBuffer create_global_circular_buffer(
    MeshDevice* mesh_device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

// DRAM-sender variant: senders are programmable DRAM cores identified by DRAM bank id.
// The returned GlobalCircularBuffer is the same type as the worker variant; the sender
// domain is queryable via tt::tt_metal::experimental::sender_core_type(gcb).
GlobalCircularBuffer create_global_circular_buffer_for_tensor_prefetcher(
    MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type = BufferType::L1,
    bool support_multi_receiver_shards = true);

// Build a DRAM-sender GCB shaped to feed one or more 1D ring matmuls (gather_in0=true) from the
// given weight tensors. The caller supplies `bank_to_receivers` (the same layout the low-level
// `create_global_circular_buffer_for_tensor_prefetcher` takes); this factory validates it against
// the matmul program configs and weight shapes and sizes the GCB accordingly.
//
// The DRAM layout is detected from the weight allocation, NOT chosen by the caller — the tensor's
// layout determines what the prefetcher does (mirrors tt_metal detect_layout_mode). All weights
// sharing one GCB must use the same layout:
//   * Legacy K-row-major (WIDTH_SHARDED): one shard per bank, K-row-major within the bank, so one
//     read serves all of a bank's receivers. Always single-sender per bank. Validated: weight K
//     tile-aligned AND divisible by ring_size (the silent-hang case where the matmul pads K beyond
//     what the prefetcher pushes); weight N shards evenly across senders and per-bank N splits
//     evenly across receivers; matmul per_core_N == weight per-receiver N. `bank_to_receivers`
//     shape is checked: num_senders * num_global_cb_receivers == ring_size, every bank owns exactly
//     num_global_cb_receivers cores. `size` floor is one full layer (ring_size * largest in1 block).
//   * Receiver-contiguous (NdShardSpec): num_shards == ring_size, each shard (full K, N/ring_size)
//     owned by one receiver. Validated: num_shards == ring_size, weight K-tiles divisible by
//     ring_size, per_core_N == per-receiver N. Receivers need NOT be uniform per bank (strided
//     round-robin). `size` floor is ring_size * largest per-receiver page, relaxed to a
//     double-buffer window when every config is stream_in1. This layout also permits dual senders
//     per bank via `support_multi_receiver_shards=false` (see below).
//
// All configs must agree on compute_with_storage_grid_size (and, for legacy, num_global_cb_receivers)
// — the GCB has one receiver rectangle shared across all consumer matmuls. The factory does NOT
// check that receivers row-major-walk into the matmul's activation-grid order (the matmul op asserts
// that at construction time), nor L1 capacity — callers must size the GCB to fit alongside the
// matmul's in0/in1/out/interm CBs on the receiver cores. `size` is capped against a conservative
// remote-CB page-count limit; larger-than-one-layer buffering does not raise throughput (the DRISC
// stalls on remote_cb_reserve_back).
//
// `support_multi_receiver_shards` is an optional override for the per-bank sender count; leave it
// unset (nullopt) in production — the sender count is derived from the detected layout:
//   * legacy K-row-major -> single sender per bank (the only option);
//   * receiver-contiguous -> DUAL senders per bank (the higher-bandwidth default; single-receiver
//     banks fall back to one sender automatically).
// An explicit value overrides that default, mainly for tests/benchmarks: true forces single sender,
// false forces dual senders. Recall the sense — true = a bank's shard may feed multiple receivers
// (single sender); false = each receiver owns its own shard (dual senders). Forcing false (dual) on
// a legacy weight is an error (that layout is inherently single-sender per bank).
GlobalCircularBuffer create_global_circular_buffer_for_matmul_1d(
    MeshDevice* mesh_device,
    const std::vector<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>& program_configs,
    const std::vector<tt::tt_metal::Tensor>& weights,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type = BufferType::L1,
    std::optional<bool> support_multi_receiver_shards = std::nullopt);

// Compute the validated `block_count` to pair with `weight` in a TensorPrefetcherInput when feeding a
// gather_in0 1D matmul (`program_config`) from a *receiver-contiguous* DRAM weight via `gcb`. This is the
// single place that owns the recv-contig prefetcher↔matmul cross-checks that otherwise have to be
// reproduced (and have drifted) at every call site. It validates, then returns `block_count`:
//   * block_count == ring_size (the matmul does wait_front(ring_size) per layer);
//   * the weight is an NdShardSpec DRAM tensor whose num_shards == ring_size (one shard per receiver),
//     each shard spanning the full K and N/ring_size columns;
//   * weight K-tiles is divisible by ring_size — otherwise the prefetcher ceil-rounds the K-block width
//     and over-reads past the receiver's slab while the matmul waits on pages that never come;
//   * the weight's per-receiver N (shard N in tiles) equals program_config.per_core_N — otherwise the
//     prefetcher's pushed page size and the matmul's in1 remote-CB page size disagree and the page-credit
//     accounting desyncs.
// Throws (TT_FATAL) on any mismatch. Mirrors create_global_circular_buffer_for_matmul_1d's K-row-major
// guards for the receiver-contiguous layout.
uint32_t tensor_prefetcher_block_count_for_matmul_1d(
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
    const tt::tt_metal::Tensor& weight,
    const GlobalCircularBuffer& gcb);

}  // namespace ttnn::global_circular_buffer
