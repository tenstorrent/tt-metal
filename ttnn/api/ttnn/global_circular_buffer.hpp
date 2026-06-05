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
    BufferType buffer_type = BufferType::L1);

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

}  // namespace ttnn::global_circular_buffer
