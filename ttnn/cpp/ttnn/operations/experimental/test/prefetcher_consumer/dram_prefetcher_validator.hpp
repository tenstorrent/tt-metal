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
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::test {

// Byte-for-byte validator companion to ttnn.dram_prefetcher / start_tensor_prefetcher.
// Loads a validator kernel on each receiver core of the supplied GCB; for each pushed
// page it reads the receiver's expected tile range from `source_tensor` (via
// TensorAccessor) and memcmps against the received bytes.
struct DramPrefetcherValidatorDeviceOperation {
    struct operation_attributes_t {
        uint32_t num_layers;
        uint32_t print_stride;
        // optional<> because reflection-based profiler serialization needs a default-
        // constructible attribute struct, and GlobalCircularBuffer has no default ctor.
        std::optional<tt::tt_metal::experimental::GlobalCircularBuffer> global_cb;
    };

    struct tensor_args_t {
        const ttnn::Tensor& source_tensor;
    };

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
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

// Public free function (kept for the nanobind binding `ttnn.experimental.test_dram_prefetcher_validator`).
void test_dram_prefetcher_validator(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const ttnn::Tensor& source_tensor,
    uint32_t num_layers,
    uint32_t print_stride,
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb);

}  // namespace ttnn::operations::experimental::test
