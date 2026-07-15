// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
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
GlobalCircularBuffer create_global_circular_buffer_with_dram_senders(
    MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type = BufferType::L1,
    bool support_multi_receiver_shards = true);

// Build a DRAM-sender GCB shaped to feed one or more 1D ring matmuls (gather_in0=true)
// from the given weight tensors. The caller supplies `bank_to_receivers` (the same layout
// the low-level `create_global_circular_buffer_with_dram_senders` takes), and the factory
// validates it is compatible with the matmul program configs and weight shapes.
//
// One (program_config, weight) pair per matmul. Each pair is validated independently:
//   * weight K is tile-aligned AND divisible by ring_size (so activation K_per_shard is
//     integer-tile, the silent-hang case where matmul pads K beyond what the prefetcher
//     pushes),
//   * weight N shards evenly across senders (num_senders = bank_to_receivers.size()) and
//     per-bank N splits evenly across receivers,
//   * matmul's per_core_N matches the weight per-receiver N.
// All configs must agree on compute_with_storage_grid_size and num_global_cb_receivers
// (the GCB has one receiver rectangle shared across all consumer matmuls).
//
// `bank_to_receivers` shape is checked: num_senders * num_global_cb_receivers must equal
// ring_size, and every bank must own exactly num_global_cb_receivers cores. The factory
// does NOT check that the receivers row-major-walk into the same order as the matmul's
// activation grid — the matmul op asserts that at construction time.
//
// `size` is the GCB size in bytes. Picking it:
//   * Minimum = num_blocks * largest_in1_block_size (one full layer's pages). The matmul
//     does wait_front(num_blocks) per layer, so anything smaller deadlocks.
//   * Larger values let the DRISC prefetcher run further ahead. Past one full layer worth
//     more buffering does not increase throughput — the DRISC stalls on
//     remote_cb_reserve_back.
//   * The factory only caps `size` against a conservative remote-CB page-count limit; it
//     does NOT check L1 capacity. Callers must size the GCB to fit alongside the matmul's
//     in0/in1/out/interm CBs on the receiver cores.
GlobalCircularBuffer create_global_circular_buffer_for_matmul_1d(
    MeshDevice* mesh_device,
    const std::vector<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>& program_configs,
    const std::vector<tt::tt_metal::Tensor>& weights,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

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

// Receiver-contiguous counterpart of create_global_circular_buffer_for_matmul_1d: build a DRAM-sender
// GCB sized to feed one or more gather_in0 1D ring matmuls from *receiver-contiguous* (NdShardSpec) DRAM
// weights, and validate the (program_config, weight, bank_to_receivers) triple in one place.
//
// Differences from the K-row-major builder, all inherent to the recv-contig layout:
//   * `size` minimum is ring_size * largest per-receiver page (= (K_tiles/ring_size) * per_core_N * tile);
//   * receivers need NOT be uniform per bank and the bank->ring mapping is the strided round-robin one,
//     so no per-bank-count / contiguous-ring assertions (the matmul op asserts the strided walk);
//   * clearing `support_multi_receiver_shards` (i.e. promising each receiver owns a disjoint
//     contiguous shard) lets each bank split its receivers across two DRISC sender cores
//     (a single-receiver bank falls back to one sender, so mixed single/dual banks are allowed).
//     The Tensor prefetcher always provisions both cores and targets the subset mapped by this GCB.
//
// Per (config, weight) it runs the same recv-contig cross-checks as
// tensor_prefetcher_block_count_for_matmul_1d (num_shards == ring_size, weight K-tiles divisible by
// ring_size, per_core_N == per-receiver N). Throws (TT_FATAL) on any mismatch.
GlobalCircularBuffer create_global_circular_buffer_for_matmul_1d_recv_contig(
    MeshDevice* mesh_device,
    const std::vector<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>& program_configs,
    const std::vector<tt::tt_metal::Tensor>& weights,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type = BufferType::L1,
    bool support_multi_receiver_shards = true);

}  // namespace ttnn::global_circular_buffer
