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
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::unary {

struct UnaryDeviceOperation {
    using spec_return_value_t = TensorSpec;
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

        tt::stl::hash::hash_t to_hash() const;
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
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);

    // The reader/writer/compute per-core args (work-split tile counts + start ids) and the baked SFPU
    // scalars are SHAPE/SCALAR-derived but are NOT all covered by compute_program_hash (for TILED
    // interleaved padded_shape is excluded). On a cache hit the descriptor is not rebuilt, so the args
    // baked at first miss would otherwise stay frozen and corrupt a differently-shaped call sharing the
    // same cache entry. Re-derive and re-apply them on every dispatch here (the tensor addresses are
    // patched separately via the Buffer* arg-0 bindings).
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
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
