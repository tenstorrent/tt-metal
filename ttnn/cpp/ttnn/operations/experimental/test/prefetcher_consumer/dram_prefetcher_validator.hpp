// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/tensor_prefetcher/tensor_prefetcher.hpp"
#include "ttnn/prefetcher_pipe.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::test {

// Byte-for-byte validator companion to ttnn.dram_prefetcher / start_tensor_prefetcher.
// Loads a validator kernel on each receiver core of the supplied delivery target, a GCB or
// PrefetcherPipes; for each pushed page it reads the receiver's expected tile range from
// `source_tensor` (via TensorAccessor) and memcmps against the received bytes.
struct DramPrefetcherValidatorDeviceOperation {
    struct operation_attributes_t {
        uint32_t num_layers;
        uint32_t print_stride;
        // Exactly one delivery target is set: global_cb, or a non-empty prefetcher_pipes. optional<>
        // because reflection-based profiler serialization needs a default-constructible attribute
        // struct, and GlobalCircularBuffer has no default ctor.
        std::optional<tt::tt_metal::experimental::GlobalCircularBuffer> global_cb;
        ttnn::PrefetcherPipeList prefetcher_pipes;
        // With prefetcher_pipes: the plan the pipes were created from
        // (create_prefetcher_pipes_for_tensor_prefetcher). It fixes which slab each receiver expects,
        // so the pipes may be in any order.
        std::vector<std::pair<uint32_t, tt::tt_metal::CoreRangeSet>> bank_to_receivers;
        // When true, expect the streaming prefetcher's ring-rotated delivery: the block at FIFO
        // position p for a receiver is physical block (lead_block + p) mod num_blocks of its slab,
        // consumed one page at a time. False = batched natural order 0..N-1.
        bool streaming = false;
        // Per-receiver streaming rotation, indexed by global ring position (must match what was
        // queued to the prefetcher). Empty == identity (lead_block = ring_pos), the natural
        // topology order. Non-empty exercises a host-chosen lead block per receiver.
        std::vector<uint32_t> rotation;
    };

    struct tensor_args_t {
        const ttnn::Tensor& source_tensor;
    };

    // Side-effect op (no output tensors).
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<ttnn::Tensor>;

    // The GlobalCircularBuffer target: a remote CB the validator kernel reads.
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

    // The PrefetcherPipe target: a Metal 2.0 spec factory whose program binds the pipes and the source
    // tensor as parameters, which a cache hit rebinds from prefetcher_pipes and source_tensor.
    struct PrefetcherPipeSpecFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ProgramFactory, PrefetcherPipeSpecFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

// Public free function (kept for the nanobind binding `ttnn.experimental.test_dram_prefetcher_validator`).
void test_dram_prefetcher_validator(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const ttnn::Tensor& source_tensor,
    uint32_t num_layers,
    uint32_t print_stride,
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb,
    bool streaming = false,
    const std::vector<uint32_t>& rotation = {});

// Same validation against a PrefetcherPipe target (bound as
// `ttnn.experimental.test_tensor_prefetcher_pipe_validator`). `streaming` / `rotation` mean what
// they do above and must match what was queued; the consumer Attaches the pipes and reads them
// through the device-side experimental::PrefetcherPipe. `bank_to_receivers` is the plan the pipes
// were created from (create_prefetcher_pipes_for_tensor_prefetcher); it fixes which slab each
// receiver expects, so `prefetcher_pipes` may be in any order. Returns once the validator has run
// on the calling thread's command queue.
void test_tensor_prefetcher_pipe_validator(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const ttnn::Tensor& source_tensor,
    uint32_t num_layers,
    uint32_t print_stride,
    const std::vector<std::shared_ptr<tt::tt_metal::experimental::PrefetcherPipe>>& prefetcher_pipes,
    const std::vector<std::pair<uint32_t, tt::tt_metal::CoreRangeSet>>& bank_to_receivers,
    bool streaming = false,
    const std::vector<uint32_t>& rotation = {});

// The program test_tensor_prefetcher_pipe_validator launches, as a MeshWorkload for a caller that enqueues it
// some other way, such as recording it into a CommandList. Each call builds a new program; the program cache is
// not consulted.
tt::tt_metal::distributed::MeshWorkload make_tensor_prefetcher_pipe_validator_workload(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const ttnn::Tensor& source_tensor,
    uint32_t num_layers,
    uint32_t print_stride,
    const std::vector<std::shared_ptr<tt::tt_metal::experimental::PrefetcherPipe>>& prefetcher_pipes,
    const std::vector<std::pair<uint32_t, tt::tt_metal::CoreRangeSet>>& bank_to_receivers,
    bool streaming = false,
    const std::vector<uint32_t>& rotation = {});

}  // namespace ttnn::operations::experimental::test
