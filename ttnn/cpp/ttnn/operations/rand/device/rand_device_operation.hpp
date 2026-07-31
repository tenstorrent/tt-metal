// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <variant>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

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

        // from/to/seed ride in as runtime args, re-applied per dispatch, so keying them would compile a
        // program per seed. mesh_dim_is_sharded only shifts the per-device seed.
        static constexpr auto attributes_excluded_from_key =
            std::forward_as_tuple("from", "to", "seed", "mesh_dim_is_sharded");
    };

    struct tensor_args_t {};

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    // Metal 2.0 spec factory (CustomProgramSpecFactoryConcept): create_program_artifacts returns a
    // ProgramSpec + ProgramRunArgs; override_runtime_arguments re-applies the DYNAMIC seed/from/to
    // runtime args (omitted from the cache key) on every cache hit via UpdateProgramRunArgs, so calls
    // differing only in seed/from/to reuse the cached program instead of recompiling. The seed/from/to
    // values built in create_program_artifacts must mirror those in override_runtime_arguments — the
    // test_rand_different_seed_values regression test enforces this.
    struct RandProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);

        static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };
    using program_factory_t = std::variant<RandProgramFactory>;

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
