// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <variant>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/metal2_artifacts.hpp"

namespace ttnn::operations::rand {

struct RandDeviceOperation {
    struct operation_attributes_t {
        const ttnn::Shape shape;
        DataType dtype;
        Layout layout;
        const MemoryConfig memory_config;
        MeshDevice* device;
        const float from;
        const float to;
        uint32_t seed;
        ttsl::SmallVector<bool> mesh_dim_is_sharded;

        // Program identity. seed/from/to are re-applied via get_dynamic_runtime_args (excluded
        // here). `device` must be FIRST: rand has no input tensor, so the framework discovers the
        // mesh device via get_first_object_of_type over attribute_values(), and its tuple path only
        // inspects element 0.
        static constexpr auto attribute_names =
            std::forward_as_tuple("device", "shape", "dtype", "layout", "memory_config");
        auto attribute_values() const { return std::forward_as_tuple(device, shape, dtype, layout, memory_config); }
    };

    struct tensor_args_t {};

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    // Metal 2.0 program factory: builds the immutable ProgramSpec and its mutable
    // ProgramRunArgs. seed/from/to are excluded from the program hash (calls differing only
    // in those values cache-hit instead of recompiling); they live in ProgramRunArgs and are
    // re-applied on every dispatch by the framework adapter (cache miss and cache hit alike),
    // so a cache hit still sees a fresh per-core seed. The test_rand_different_seed_values
    // regression test enforces this.
    struct RandProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_spec(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };
    using program_factory_t = std::variant<RandProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return RandProgramFactory{};
    }

    static void validate_inputs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::rand

namespace ttnn::prim {
ttnn::operations::rand::RandDeviceOperation::tensor_return_value_t uniform(
    const ttnn::Shape& shape,
    DataType dtype,
    Layout layout,
    const MemoryConfig& memory_config,
    MeshDevice& device,
    float from,
    float to,
    uint32_t seed,
    ttsl::SmallVector<bool> mesh_dim_is_sharded = {});
}  // namespace ttnn::prim
