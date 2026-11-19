// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/ternary/common/ternary_op_types.hpp"
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt_stl/runtime_only.hpp>

namespace ttnn::operations::ternary {

struct TernaryDeviceOperation {
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct operation_attributes_t {
        TernaryOpType ternary_op_type;
        TernaryVariant ternary_variant;
        TernaryBroadcastType broadcast_type;
        tt::tt_metal::MemoryConfig memory_config;
        DataType input_dtype;
        const CoreRangeSet worker_grid;
        std::optional<DataType> dtype;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config;
        std::optional<CoreRangeSet> sub_core_grids;

        // Packed into runtime args and re-applied on every cache hit, so one program serves every
        // scalar value. Every other attribute is part of the program key (compute_program_hash_inputs).
        ttsl::runtime_only<std::optional<ScalarVariant>> scalar_input_a;
        ttsl::runtime_only<std::optional<ScalarVariant>> scalar_input_b;

        DataType get_dtype() const;
    };

    // Shape-blind program key of one tensor operand. Interleaved operands key on everything but the row
    // count, so one program serves every height (per-core args are re-derived on a hit).
    struct OperandKey {
        DataType dtype;
        tt::tt_metal::MemoryConfig memory_config;
        // (logical H == 1, logical W): a single row is broadcast along H; any other H runs the same program.
        std::pair<uint32_t, uint32_t> broadcast;
        tt::tt_metal::Alignment alignment;
        tt::tt_metal::Tile tile;
        // Sharded only: common runtime args bake the shape in pages and are not rewritten on a hit.
        std::optional<tt::tt_metal::Shape> shard_shape_in_pages;
    };

    struct OutputKey {
        DataType dtype;
        tt::tt_metal::Alignment alignment;
        tt::tt_metal::Tile tile;
        std::optional<tt::tt_metal::Shape> shard_shape_in_pages;
    };

    // Shard volume in tiles of (predicate, true, false, output) when native sharding applies.
    using ShardVolumesKey = std::optional<std::array<std::optional<uint32_t>, 4>>;

    // Everything the cached program depends on. The framework derives both the program hash and the exact
    // canonical cache key from this tuple. All attributes are included (the runtime_only scalars are skipped),
    // so a new attribute is keyed by default. An absent b / c operand (nullopt) distinguishes TTS/TST from TTT.
    using program_hash_inputs_t = std::tuple<
        const operation_attributes_t&,
        OperandKey,
        std::optional<OperandKey>,
        std::optional<OperandKey>,
        ShardVolumesKey,
        OutputKey>;

    struct tensor_args_t {
        const Tensor& input_tensor_a;
        std::optional<Tensor> input_tensor_b;
        std::optional<Tensor> input_tensor_c;
        std::optional<Tensor> optional_output_tensor;
    };

    struct TernaryProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);

        // Cache-hit re-apply of ALL per-core runtime args and every tensor binding -- no program
        // rebuild and no kernel recompile, but the work split and every shape-dependent arg is
        // re-derived for the current tensors, via the same shared builder
        // create_program_artifacts() uses.
        static tt::tt_metal::ProgramRunArgs override_runtime_arguments(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<TernaryProgramFactory>;
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static program_hash_inputs_t compute_program_hash_inputs(const operation_attributes_t&, const tensor_args_t&);
    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);
};

}  // namespace ttnn::operations::ternary

namespace ttnn::prim {

ttnn::operations::ternary::TernaryDeviceOperation::tensor_return_value_t ternary(
    ttnn::operations::ternary::TernaryOpType op_type,
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

ttnn::operations::ternary::TernaryDeviceOperation::tensor_return_value_t ternary(
    ttnn::operations::ternary::TernaryOpType op_type,
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    ttnn::operations::ternary::ScalarVariant scalar,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

ttnn::operations::ternary::TernaryDeviceOperation::tensor_return_value_t ternary(
    ttnn::operations::ternary::TernaryOpType op_type,
    const Tensor& input_a,
    const Tensor& input_b,
    float scalar_c,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

ttnn::operations::ternary::TernaryDeviceOperation::tensor_return_value_t ternary(
    ttnn::operations::ternary::TernaryOpType op_type,
    const Tensor& input_a,
    float scalar_b,
    const Tensor& input_c,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn::prim
