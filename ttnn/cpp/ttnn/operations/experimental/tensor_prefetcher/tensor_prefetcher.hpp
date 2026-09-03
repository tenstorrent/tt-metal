// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal {
class Program;
namespace distributed {
class MeshDevice;
}
}  // namespace tt::tt_metal

namespace ttnn::operations::experimental {

// One Tensor-prefetcher delivery target: the PrefetcherPipes of every DRAM sender core, all
// sharing one entry size and ring geometry. A PrefetcherPipe is one sender and a bank may be
// driven by two of them, so a target that spans the DRAM banks is a per-bank group of pipes; this
// bundles those groups with the geometry every pipe in them agrees on, which is what the
// prefetcher and the consumer op both name.
//
// Order is semantic and is the one CreatePrefetcherPipesForTensorPrefetcher returned: within a
// bank, the first pipe owns that bank's leading receivers (bank-local slab index 0). Everything
// that enumerates pipes here does so bank-major, so a pipe's position in `attach()`'s ids matches
// its position in `sender_receiver_core_mapping()`.
//
// Copyable: the pipes are shared. Keep a copy alive for as long as any program has Attached them
// or the prefetcher may still deliver into them -- an attached Program holds a non-owning pointer
// to each pipe, and dropping the last copy frees the rings and config pages.
struct TensorPrefetcherPipes {
    std::vector<tt::tt_metal::experimental::TensorPrefetcherBankPipes> banks;
    // Per-receiver push granularity, shared by every pipe. Every Attach and every queued tensor
    // must match it: a DRAM sender is never dispatched to and so cannot answer a resize.
    uint32_t entry_size = 0;
    // Entries a receiver's ring holds.
    uint32_t num_entries = 0;

    // Bytes of ring per receiver.
    uint32_t ring_size() const { return entry_size * num_entries; }
    uint32_t num_banks() const { return static_cast<uint32_t>(banks.size()); }
    // Pipes across every bank, which is one per DRAM sender core.
    uint32_t num_pipes() const;
    // Every receiver across every pipe. This is the core set a consumer program attaches.
    CoreRangeSet receiver_cores() const;
    // Sender core (DRAM-logical, x == bank id) -> its receivers. One entry per pipe, bank-major.
    std::vector<std::pair<tt::tt_metal::CoreCoord, CoreRangeSet>> sender_receiver_core_mapping() const;
    // Attach every pipe to `program` on its own receiver cores, at the shared entry size. Returns
    // the program-local pipe id per pipe, bank-major; a receiver core's kernel takes the id of
    // the one pipe it belongs to (as a runtime argument, since one kernel serves receivers of
    // different pipes).
    std::vector<uint8_t> attach(tt::tt_metal::Program& program) const;
};

// One tensor to prefetch: either (tensor, block_count) or (tensor, block_count, rotation).
// block_count is the number of K-blocks to divide the tensor's K dimension into. rotation
// (receiver-contiguous layout only; omitted/empty == batched) is the per-receiver streaming
// ring-rotation table, indexed by global ring position: it delivers that tensor's K-blocks in
// host-specified ring-rotated FIFO order for a matching stream_in1 matmul. See
// TensorPrefetcherInput for the rotation contract.
using TensorPrefetcherQueueTensor =
    std::variant<std::pair<ttnn::Tensor, uint32_t>, std::tuple<ttnn::Tensor, uint32_t, std::vector<uint32_t>>>;

// Thin ttnn-side wrappers around the queueable
// tt::tt_metal::experimental::Start/Queue/Stop TensorPrefetcher API.
//
// Lifecycle:
//   1. start_tensor_prefetcher(device)
//      - Builds the long-running DRISC kernel on every DRAM sender core and
//        spawns the host worker thread. Returns immediately. Receiver count
//        is per-GCB (read from each GCB's sender state block on every
//        request), so a single prefetcher can serve GCBs with different
//        num_receivers values.
//   2. queue_tensor_prefetcher_request(device, tensors, global_cb, device_subset=None,
//                                      capture_into_trace=False)
//      - Push one request. `tensors` is the full, flattened list of weights (at
//        least one), streamed in list order; each item is (weight, block_count)
//        or (weight, block_count, rotation) (see TensorPrefetcherQueueTensor).
//        block_count is the number of K-blocks to divide that tensor's K
//        dimension into. Pass distinct tensors for distinct layers, or repeat a
//        tensor to replay it. device_subset defaults to the full mesh. Set
//        capture_into_trace=True to let the request be captured into a trace being
//        recorded on the current command queue; by default it is always sent
//        immediately and never captured.
//   3. stop_tensor_prefetcher(device)
//      - Sends the stop sentinel, joins the worker, waits for the kernels
//        to exit. Caller must call this before destroying the device.
// Returns true if the Tensor prefetcher is supported on `mesh_device`
// (programmable DRAM cores are available). Use this to skip rather than fail
// when start_tensor_prefetcher would otherwise raise.
bool is_tensor_prefetcher_supported(tt::tt_metal::distributed::MeshDevice* mesh_device);

void start_tensor_prefetcher(tt::tt_metal::distributed::MeshDevice* mesh_device);

// `capture_into_trace` selects whether this request may be captured into a trace: when true
// and the calling thread's current command queue is mid trace-capture, the request is captured
// and re-sent on every execute_trace of that trace; when false the request is always sent
// immediately.
// Exactly one of `global_cb` / `prefetcher_pipes` must be supplied; whichever it is selects the
// delivery transport. See the metal-level QueueTensorPrefetcherRequest overloads for the extra
// preconditions PrefetcherPipe delivery imposes (receiver-contiguous, batched, fixed entry size).
void queue_tensor_prefetcher_request(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const std::vector<TensorPrefetcherQueueTensor>& tensors,
    const std::optional<tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb = std::nullopt,
    const std::optional<TensorPrefetcherPipes>& prefetcher_pipes = std::nullopt,
    const std::optional<tt::tt_metal::distributed::MeshCoordinateRangeSet>& device_subset = std::nullopt,
    bool capture_into_trace = false);

// Create the PrefetcherPipes whose senders are programmable DRAM cores, as a delivery target for
// the Tensor prefetcher. Sender placement matches create_global_circular_buffer_for_tensor_prefetcher.
TensorPrefetcherPipes create_prefetcher_pipes_for_tensor_prefetcher(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t entry_size,
    uint32_t num_entries,
    tt::tt_metal::BufferType buffer_type = tt::tt_metal::BufferType::L1,
    bool support_multi_receiver_shards = false);

// Fence the prefetcher against a command queue: every prefetch request queued after this
// call waits until all work previously enqueued on that queue has completed on device before
// the prefetcher reads DRAM. Call after the data writes and before the dependent
// queue_tensor_prefetcher_request. `cq_id` defaults to the calling thread's current queue.
void wait_for_cq_on_tensor_prefetcher(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    std::optional<uint8_t> cq_id = std::nullopt,
    const std::optional<tt::tt_metal::distributed::MeshCoordinateRangeSet>& device_subset = std::nullopt);

void stop_tensor_prefetcher(tt::tt_metal::distributed::MeshDevice* mesh_device);

}  // namespace ttnn::operations::experimental
