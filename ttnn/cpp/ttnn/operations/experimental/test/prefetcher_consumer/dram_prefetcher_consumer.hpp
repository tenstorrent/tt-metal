// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/tensor_prefetcher/tensor_prefetcher.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::test {

// Bench-only companion to the Tensor prefetcher. Loads a discard-only receiver kernel on each
// receiver core of the supplied delivery target; each receiver runs `wait_front(1); pop_front(1);`
// in a loop `num_iters` times. Which target is supplied selects the transport, the same way
// queue_tensor_prefetcher_request does, so one bench can measure either.
struct DramPrefetcherConsumerDeviceOperation {
    struct operation_attributes_t {
        uint32_t num_iters;
        // Per-receiver bytes one wait_front/pop_front covers: the GCB's receiver page size, or the
        // entry size the PrefetcherPipes are Attached at. Either way it must be what the sender
        // pushes per receiver per block.
        uint32_t page_size_bytes;
        // Exactly one of these is set. optional<> because reflection-based profiler serialization
        // needs a default-constructible attribute struct, and GlobalCircularBuffer has no default
        // ctor.
        std::optional<tt::tt_metal::experimental::GlobalCircularBuffer> global_cb;
        std::optional<ttnn::operations::experimental::TensorPrefetcherPipes> prefetcher_pipes;
        ttnn::MeshDevice* mesh_device;
    };

    struct tensor_args_t {};

    // Side-effect op (no output tensors).
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<ttnn::Tensor>;

    struct ProgramFactory {
        struct shared_variables_t {};
        using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

        static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinate& mesh_coordinate,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

// Public free function (kept for the nanobind binding `ttnn.experimental.test_dram_prefetcher_consumer`).
void test_dram_prefetcher_consumer(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    uint32_t num_iters,
    uint32_t page_size_bytes,
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb);

// Same discard-only drain against a PrefetcherPipe target (bound as
// `ttnn.experimental.test_tensor_prefetcher_pipe_consumer`), so the two transports can be benched
// head to head. `num_iters` counts entries per receiver, and `page_size_bytes` is the entry size to
// Attach at.
void test_tensor_prefetcher_pipe_consumer(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    uint32_t num_iters,
    uint32_t page_size_bytes,
    const ttnn::operations::experimental::TensorPrefetcherPipes& prefetcher_pipes);

}  // namespace ttnn::operations::experimental::test
