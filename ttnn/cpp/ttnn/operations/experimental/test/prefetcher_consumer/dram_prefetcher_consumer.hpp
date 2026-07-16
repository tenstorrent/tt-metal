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
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::test {

// Bench-only companion to `ttnn.dram_prefetcher`. Loads a discard-only receiver kernel
// on each receiver core of the supplied GCB; each receiver runs `wait_front(1);
// pop_front(1);` in a loop `num_iters` times.
struct DramPrefetcherConsumerDeviceOperation {
    struct operation_attributes_t {
        uint32_t num_iters;
        uint32_t page_size_bytes;
        // optional<> because reflection-based profiler serialization needs a default-
        // constructible attribute struct, and GlobalCircularBuffer has no default ctor.
        std::optional<tt::tt_metal::experimental::GlobalCircularBuffer> global_cb;
        ttnn::MeshDevice* mesh_device;

        static constexpr auto attribute_names =
            std::forward_as_tuple("num_iters", "page_size_bytes", "global_cb_config_address");
        auto attribute_values() const {
            return std::make_tuple(
                num_iters,
                page_size_bytes,
                global_cb.has_value() ? static_cast<uint64_t>(global_cb->config_address()) : uint64_t{0});
        }
    };

    struct tensor_args_t {};

    // Side-effect op (no output tensors).
    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
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
};

// Public free function (kept for the nanobind binding `ttnn.experimental.test_dram_prefetcher_consumer`).
void test_dram_prefetcher_consumer(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    uint32_t num_iters,
    uint32_t page_size_bytes,
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb);

}  // namespace ttnn::operations::experimental::test
