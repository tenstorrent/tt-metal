// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_prefetcher.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/experimental/tensor_prefetcher.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace ttnn::operations::experimental {

namespace metal_exp = tt::tt_metal::experimental;

TensorPrefetcherPipes::TensorPrefetcherPipes(
    std::vector<tt::tt_metal::experimental::TensorPrefetcherBankPipes> banks,
    uint32_t entry_size,
    uint32_t num_entries) :
    banks(std::move(banks)),
    entry_size(entry_size),
    num_entries(num_entries),
    mapping_(metal_exp::prefetcher_pipe_sender_receiver_mapping(this->banks)) {
    config_addresses_.reserve(mapping_.size());
    for (const auto& bank : this->banks) {
        for (const auto& pipe : bank.pipes) {
            config_addresses_.push_back(metal_exp::prefetcher_pipe_config_address(*pipe));
        }
    }
}

CoreRangeSet TensorPrefetcherPipes::receiver_cores() const {
    // One merge over every pipe's ranges: CoreRangeSet::merge rasterizes the whole bounding box,
    // so folding pipe by pipe would redo that work per pipe.
    std::vector<tt::tt_metal::CoreRange> ranges;
    for (const auto& [_sender, receivers] : mapping_) {
        ranges.insert(ranges.end(), receivers.ranges().begin(), receivers.ranges().end());
    }
    return CoreRangeSet().merge(ranges);
}

std::vector<uint8_t> TensorPrefetcherPipes::attach(tt::tt_metal::Program& program) const {
    std::vector<uint8_t> pipe_ids;
    pipe_ids.reserve(num_pipes());
    for (const auto& bank : banks) {
        for (const auto& pipe : bank.pipes) {
            pipe_ids.push_back(metal_exp::AttachPrefetcherPipe(
                program, *pipe, metal_exp::prefetcher_pipe_receiver_cores(*pipe), entry_size));
        }
    }
    return pipe_ids;
}

bool is_tensor_prefetcher_supported(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    return tt::tt_metal::experimental::IsTensorPrefetcherSupported(*mesh_device);
}

void start_tensor_prefetcher(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    tt::tt_metal::experimental::StartTensorPrefetcher(*mesh_device, {});
}

void queue_tensor_prefetcher_request(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const std::vector<TensorPrefetcherQueueTensor>& tensors,
    const std::optional<tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const std::optional<TensorPrefetcherPipes>& prefetcher_pipes,
    const std::optional<tt::tt_metal::distributed::MeshCoordinateRangeSet>& device_subset,
    bool capture_into_trace) {
    const bool has_gcb = global_cb.has_value();
    const bool has_pipes = prefetcher_pipes.has_value() && !prefetcher_pipes->banks.empty();
    TT_FATAL(
        has_gcb != has_pipes,
        "queue_tensor_prefetcher_request needs exactly one delivery target: global_cb {} supplied, "
        "prefetcher_pipes {} supplied",
        has_gcb ? "was" : "was not",
        has_pipes ? "was" : "was not");

    std::vector<tt::tt_metal::experimental::TensorPrefetcherInput> inputs;
    inputs.reserve(tensors.size());
    for (const auto& item : tensors) {
        // (tensor, block_count) defaults to batched (empty rotation); (tensor, block_count,
        // rotation) supplies the per-receiver streaming rotation table for that tensor.
        if (const auto* pair = std::get_if<std::pair<ttnn::Tensor, uint32_t>>(&item)) {
            inputs.push_back({pair->first.mesh_tensor(), pair->second, /*rotation=*/{}});
        } else {
            const auto& [tensor, block_count, rotation] =
                std::get<std::tuple<ttnn::Tensor, uint32_t, std::vector<uint32_t>>>(item);
            inputs.push_back({tensor.mesh_tensor(), block_count, rotation});
        }
    }
    // There is no cq_id parameter to consult: a `cq_id`/`queue_id` keyword is consumed by
    // ttnn's operation wrapper before this function runs, and applied by making that queue
    // the thread's current one. So the knob left here is whether to consider a queue at all
    // — with capture_into_trace false we hand metal no queue, and the request is sent
    // immediately even mid trace-capture.
    auto* trace_cq = capture_into_trace ? &mesh_device->mesh_command_queue() : nullptr;
    if (has_pipes) {
        tt::tt_metal::experimental::QueueTensorPrefetcherRequest(
            *mesh_device, prefetcher_pipes->banks, device_subset, inputs, trace_cq);
    } else {
        tt::tt_metal::experimental::QueueTensorPrefetcherRequest(
            *mesh_device, *global_cb, device_subset, inputs, trace_cq);
    }
}

TensorPrefetcherPipes create_prefetcher_pipes_for_tensor_prefetcher(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t entry_size,
    uint32_t num_entries,
    tt::tt_metal::BufferType buffer_type,
    bool support_multi_receiver_shards) {
    return TensorPrefetcherPipes{
        metal_exp::CreatePrefetcherPipesForTensorPrefetcher(
            *mesh_device, bank_to_receivers, entry_size, num_entries, buffer_type, support_multi_receiver_shards),
        entry_size,
        num_entries};
}

void wait_for_cq_on_tensor_prefetcher(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    std::optional<uint8_t> cq_id,
    const std::optional<tt::tt_metal::distributed::MeshCoordinateRangeSet>& device_subset) {
    // cq_id must stay optional, not `uint8_t = 0`: only the positional form reaches here as a
    // value, since a keyword cq_id= is consumed by the wrapper and applied by making that
    // queue current. Resolving nullopt to the thread's current queue is what makes the two
    // forms agree; defaulting to 0 would silently fence queue 0 for the keyword form.
    tt::tt_metal::experimental::WaitForCqOnTensorPrefetcher(mesh_device->mesh_command_queue(cq_id), device_subset);
}

void stop_tensor_prefetcher(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    tt::tt_metal::experimental::StopTensorPrefetcher(*mesh_device);
}

}  // namespace ttnn::operations::experimental
