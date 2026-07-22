// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::unary {

struct UnaryDeviceOperation {
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct operation_attributes_t {
        std::vector<unary::EltwiseUnaryWithParam> op_chain;
        DataType output_dtype;
        tt::tt_metal::MemoryConfig memory_config;
        bool fp32_dest_acc_en = false;
        bool preserve_fp32_precision = false;
        bool bfp8_pack_precise = false;
        const CoreRangeSet worker_grid;
        std::optional<CoreRangeSet> sub_core_grids;

        ttsl::hash::hash_t to_hash() const;
    };

    struct tensor_args_t {
        const Tensor& input;
        std::optional<Tensor> output_tensor;
    };

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);

    // Cache-hit re-apply of all per-dispatch state (per-core args + tensor-backed CB/buffer addresses),
    // since the hash excludes volume. See the .cpp.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::unary

namespace ttnn::prim {

Tensor unary(
    const Tensor& input,
    const std::vector<ttnn::operations::unary::EltwiseUnaryWithParam>& op_chain,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    bool fp32_dest_acc_en,
    bool preserve_fp32_precision,
    bool bfp8_pack_precise,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn::prim
